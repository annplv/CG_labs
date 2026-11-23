#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>

// загрузка и декодирование изображений 
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "geometry.hpp"
#include "scene.hpp"
#include "transform.hpp"
#include "ui.hpp"

namespace application {

namespace {

// данные шейдеров

// набор 0, binding 0 
struct SceneData {
	glm::mat4 view_proj;
	glm::vec4 camera_position;
	glm::vec4 camera_forward;
	glm::vec4 ambient;
	glm::vec4 light_direction;
	glm::vec4 light_radiance;
	glm::uvec4 light_counts;
};

// набор 0, binding 1 
struct GpuPointLight {
	glm::vec4 position;
	glm::vec4 radiance;
};

// набор 0, binding 2 
struct GpuSpotLight {
	glm::vec4 position;
	glm::vec4 direction;
	glm::vec4 radiance;
	glm::vec4 cutoff;
};

// набор 1, binding 0 : элемент динамического uniform-буфера, по одному на модель
struct ModelData {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 specular; // rgb — цвет блика, w — коэффициент блика
};

// push-константы освещённого пайплайна — включение нетривиального сэмплирования
struct LitPush {
	float time;
	uint32_t flags;
};
constexpr uint32_t flag_animated_sampling = 1u;

// push-константы для визуализации источников
struct UnlitPush {
	glm::mat4 mvp;
	glm::vec4 color;
};

// ресурсы

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

struct GpuMesh {
	Buffer vertices;
	Buffer indices;
	uint32_t index_count = 0;
};

// текстура в VRAM
struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	VkImageView view = VK_NULL_HANDLE;
};

// материал — текстуры и свой набор дескрипторов 1
// (в нём динамический uniform-буфер моделей и три пары изображение-сэмплер)
struct GpuMaterial {
	Texture albedo;
	Texture specular; // пустая, если карты нет: тогда привязывается black_texture
	Texture emissive;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

GpuMesh pyramid_mesh;
GpuMesh cube_mesh;
GpuMesh cone_mesh;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkPipelineLayout lit_layout = VK_NULL_HANDLE;
VkPipelineLayout unlit_layout = VK_NULL_HANDLE;
VkPipeline lit_pipeline = VK_NULL_HANDLE;   // пирамида с освещением
VkPipeline cube_pipeline = VK_NULL_HANDLE;  // кубики источников
VkPipeline line_pipeline = VK_NULL_HANDLE;  // конусы прожекторов
VkDescriptorPool pool = VK_NULL_HANDLE;

Buffer scene_uniforms;
Buffer point_storage;
Buffer spot_storage;
VkDescriptorSet scene_set = VK_NULL_HANDLE;

// один динамический uniform-буфер — массив ModelData для всех пирамид.
// Шаг между элементами выровнен по minUniformBufferOffsetAlignment.
Buffer model_uniforms;
VkDeviceSize model_stride = 0;

VkSampler sampler = VK_NULL_HANDLE;
Texture black_texture; // служебная текстура из одного чёрного пикселя
std::array<GpuMaterial, scene::material_count> materials;

VkCommandPool upload_pool = VK_NULL_HANDLE; // для копирования из staging-буферов в VRAM

scene::Scene world;
double previous_time = -1.0;
float current_time = 0.0f;

// буферы

// буфер в памяти, доступной CPU (uniform, storage и staging-буферы)
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo alloc = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo details = {};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &alloc, &out.handle, &out.allocation,
	                    &details) != VK_SUCCESS) {
		std::cerr << "Failed to create a buffer of " << size << " bytes\n";
		return false;
	}
	out.mapped = details.pMappedData;
	out.size = size;
	return true;
}

void upload(const Buffer& buffer, const void* data, size_t size, VkDeviceSize offset = 0) {
	if (size == 0) {
		return;
	}
	std::memcpy(static_cast<char*>(buffer.mapped) + offset, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, offset, size);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
	}
	buffer = {};
}

// Если данные не помещаются, буфер пересоздаётся вдвое большего размера.
// Безопасно: render() вызывается после prepare(), который ждёт окончания предыдущего кадра.
// Возвращает true, если буфер пересоздан.
bool growIfNeeded(Buffer& buffer, VkDeviceSize needed, VkBufferUsageFlags usage) {
	if (buffer.handle != VK_NULL_HANDLE && needed <= buffer.size) {
		return false;
	}
	const VkDeviceSize size = std::max<VkDeviceSize>({ 256, needed, buffer.size * 2 });
	destroyBuffer(buffer);
	createBuffer(size, usage, buffer);
	return true;
}

void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buffer,
                     VkDeviceSize range = VK_WHOLE_SIZE) {
	const VkDescriptorBufferInfo info = { .buffer = buffer.handle, .offset = 0, .range = range };
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// Загружает массив источников в storage-буфер; при пересоздании буфера обновляет дескриптор
template <typename T>
void uploadStorage(Buffer& buffer, uint32_t binding, const std::vector<T>& items) {
	if (growIfNeeded(buffer, items.size() * sizeof(T), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
		writeDescriptor(scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer);
	}
	upload(buffer, items.data(), items.size() * sizeof(T));
}

// Копирование в VRAM 

// Записывает команды в одноразовый буфер команд, отправляет в очередь и ждёт выполнения
template <typename Record>
bool submitOnce(Record&& record) {
	auto& ctx = graphics::internal::context;
	const VkCommandBufferAllocateInfo info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = upload_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(ctx.device, &info, &cmd) != VK_SUCCESS) {
		std::cerr << "Failed to allocate a command buffer for copying\n";
		return false;
	}

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin);
	record(cmd);
	vkEndCommandBuffer(cmd);

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	const bool ok = vkQueueSubmit(ctx.graphics_queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
	if (ok) {
		vkQueueWaitIdle(ctx.graphics_queue);
	} else {
		std::cerr << "Failed to submit copy commands\n";
	}
	vkFreeCommandBuffers(ctx.device, upload_pool, 1, &cmd);
	return ok;
}

// Буфер в VRAM: данные пишутся в staging-буфер (память CPU), затем GPU копирует их командой vkCmdCopyBuffer
bool createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
		return false;
	}
	upload(staging, data, size);

	const VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	// без флагов доступа CPU VMA выделяет память на GPU
	const VmaAllocationCreateInfo alloc = { .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
	if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &alloc, &out.handle, &out.allocation,
	                    nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create a device buffer of " << size << " bytes\n";
		destroyBuffer(staging);
		return false;
	}
	out.size = size;

	const bool ok = submitOnce([&](VkCommandBuffer cmd) {
		const VkBufferCopy region = { .size = size };
		vkCmdCopyBuffer(cmd, staging.handle, out.handle, 1, &region);
	});
	destroyBuffer(staging);
	return ok;
}

bool createMesh(const geometry::Mesh& mesh, GpuMesh& out) {
	out.index_count = uint32_t(mesh.indices.size());
	return createDeviceBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(geometry::Vertex),
	                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, out.vertices) &&
	       createDeviceBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint16_t),
	                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT, out.indices);
}

void drawMesh(VkCommandBuffer cmd, const GpuMesh& mesh) {
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertices.handle, &offset);
	vkCmdBindIndexBuffer(cmd, mesh.indices.handle, 0, VK_INDEX_TYPE_UINT16);
	vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
}

// текстуры

// Барьер, меняющий раскладку (layout) изображения
void transition(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
                VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
	const VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_access,
		.dstAccessMask = dst_access,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
	};
	vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// Текстура в VRAM из пикселей RGBA8: staging-буфер -> vkCmdCopyBufferToImage -> изображение для чтения в шейдере
bool createTexture(const void* pixels, uint32_t width, uint32_t height, VkFormat format, Texture& out) {
	auto& ctx = graphics::internal::context;
	const VkDeviceSize size = VkDeviceSize(width) * height * 4;

	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
		return false;
	}
	upload(staging, pixels, size);

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { width, height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo alloc = { .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
	if (vmaCreateImage(ctx.allocator, &image_info, &alloc, &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create a texture image " << width << "x" << height << '\n';
		destroyBuffer(staging);
		return false;
	}

	const bool ok = submitOnce([&](VkCommandBuffer cmd) {
		// UNDEFINED -> TRANSFER_DST: изображение готово принять данные
		transition(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
		           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		const VkBufferImageCopy region = {
			.imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1 },
			.imageExtent = { width, height, 1 },
		};
		vkCmdCopyBufferToImage(cmd, staging.handle, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		// TRANSFER_DST -> SHADER_READ_ONLY: копирование закончено, дальше изображение читает фрагментный шейдер
		transition(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	});
	destroyBuffer(staging);
	if (!ok) {
		return false;
	}

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = out.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
	};
	if (vkCreateImageView(ctx.device, &view_info, nullptr, &out.view) != VK_SUCCESS) {
		std::cerr << "Failed to create a texture image view\n";
		return false;
	}
	return true;
}

// Загружает изображение из файла библиотекой stb_image (всегда приводит к RGBA8) и создаёт текстуру.
// Цветные карты (альбедо, свечение) хранятся в sRGB: при чтении в шейдере GPU переводит их в линейные значения,
// в которых считается освещение. Карта блика — это данные (доля блика), поэтому UNORM.
bool loadTexture(const char* path, VkFormat format, Texture& out) {
	int width = 0;
	int height = 0;
	int channels = 0;
	stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	if (pixels == nullptr) {
		std::cerr << "Cannot load " << path << ": " << stbi_failure_reason()
		          << " (run from the project root)\n";
		return false;
	}
	const bool ok = createTexture(pixels, uint32_t(width), uint32_t(height), format, out);
	stbi_image_free(pixels);
	return ok;
}

void destroyTexture(Texture& texture) {
	auto& ctx = graphics::internal::context;
	if (texture.view != VK_NULL_HANDLE) {
		vkDestroyImageView(ctx.device, texture.view, nullptr);
	}
	if (texture.image != VK_NULL_HANDLE) {
		vmaDestroyImage(ctx.allocator, texture.image, texture.allocation);
	}
	texture = {};
}

bool createSampler() {
	const VkSamplerCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		// REPEAT: координаты вне [0, 1] повторяют текстуру (смещённые координаты при анимированном сэмплировании)
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	};
	if (vkCreateSampler(graphics::internal::context.device, &info, nullptr, &sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create the sampler\n";
		return false;
	}
	return true;
}

bool createTextures() {
	const uint8_t black[4] = { 0, 0, 0, 255 };
	if (!createSampler() || !createTexture(black, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, black_texture)) {
		return false;
	}
	for (int i = 0; i < scene::material_count; ++i) {
		const scene::MaterialTextures& files = scene::materials[i];
		GpuMaterial& m = materials[i];
		if (!loadTexture(files.albedo, VK_FORMAT_R8G8B8A8_SRGB, m.albedo) ||
		    (files.specular && !loadTexture(files.specular, VK_FORMAT_R8G8B8A8_UNORM, m.specular)) ||
		    (files.emissive && !loadTexture(files.emissive, VK_FORMAT_R8G8B8A8_SRGB, m.emissive))) {
			return false;
		}
	}
	return true;
}

void writeTextureDescriptor(VkDescriptorSet set, uint32_t binding, const Texture& texture) {
	// карты нет — привязывается чёрный пиксель: блик умножается на 0, к цвету прибавляется 0
	const VkImageView view = texture.view != VK_NULL_HANDLE ? texture.view : black_texture.view;
	const VkDescriptorImageInfo info = {
		.sampler = sampler,
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// ---------- Пайплайны ----------

VkShaderModule loadShader(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Cannot open " << path << " (run from the project root, build the shaders first)\n";
		return VK_NULL_HANDLE;
	}
	std::vector<uint32_t> code(size_t(file.tellg()) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(code.size() * sizeof(uint32_t)));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size() * sizeof(uint32_t),
		.pCode = code.data(),
	};
	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Cannot create a shader module from " << path << '\n';
	}
	return module;
}

// Пайплайны отличаются только шейдерами, раскладкой, топологией, вершинами и отсечением граней
VkPipeline createPipeline(const char* vert_path, const char* frag_path, VkPipelineLayout layout,
                          VkPrimitiveTopology topology, uint32_t stride, uint32_t attribute_count,
                          VkCullModeFlags cull) {
	auto& ctx = graphics::internal::context;

	const VkShaderModule vert = loadShader(vert_path);
	const VkShaderModule frag = loadShader(frag_path);
	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
		vkDestroyShaderModule(ctx.device, vert, nullptr);
		vkDestroyShaderModule(ctx.device, frag, nullptr);
		return VK_NULL_HANDLE;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
		  .module = vert, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		  .module = frag, .pName = "main" },
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0, .stride = stride, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	// location 0 — позиция, 1 — нормаль, 2 — текстурные координаты (нормаль и uv нужны только освещённому пайплайну)
	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, position)) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, normal)) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, uv)) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = attribute_count,
		.pVertexAttributeDescriptions = attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo raster = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = cull,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};
	const VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic,
	};

	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &raster,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth,
		.pColorBlendState = &blend,
		.pDynamicState = &dynamic_state,
		.layout = layout,
		.renderPass = ctx.render_pass,
		.subpass = 0,
	};
	VkPipeline pipeline = VK_NULL_HANDLE;
	if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create a graphics pipeline\n";
		pipeline = VK_NULL_HANDLE;
	}

	vkDestroyShaderModule(ctx.device, vert, nullptr);
	vkDestroyShaderModule(ctx.device, frag, nullptr);
	return pipeline;
}

bool createLayouts() {
	auto& ctx = graphics::internal::context;
	const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	// набор 0 (сцена): uniform-буфер + storage-буферы точечных источников и прожекторов
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
		  .stageFlags = both },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const VkDescriptorSetLayoutCreateInfo scene_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = scene_bindings,
	};

	// набор 1 (модель и материал), лаба 3: динамический uniform-буфер свойств моделей
	// и пары изображение-сэмплер для карт альбедо, блика и свечения
	const VkDescriptorSetLayoutBinding model_bindings[] = {
		{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1,
		  .stageFlags = both },
		{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
		  .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const VkDescriptorSetLayoutCreateInfo model_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 4,
		.pBindings = model_bindings,
	};

	if (vkCreateDescriptorSetLayout(ctx.device, &scene_info, nullptr, &scene_set_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(ctx.device, &model_info, nullptr, &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	// лаба 3: push-константы освещённого пайплайна — время и флаги сэмплирования для фрагментного шейдера
	const VkPushConstantRange lit_push_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(LitPush),
	};
	const VkDescriptorSetLayout lit_sets[] = { scene_set_layout, model_set_layout };
	const VkPipelineLayoutCreateInfo lit_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = lit_sets,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &lit_push_range,
	};

	// визуализации источников дескрипторы не нужны: матрица и цвет передаются push-константами
	const VkPushConstantRange push_range = { .stageFlags = both, .offset = 0, .size = sizeof(UnlitPush) };
	const VkPipelineLayoutCreateInfo unlit_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(ctx.device, &lit_info, nullptr, &lit_layout) != VK_SUCCESS ||
	    vkCreatePipelineLayout(ctx.device, &unlit_info, nullptr, &unlit_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layouts\n";
		return false;
	}
	return true;
}

bool createDescriptors() {
	auto& ctx = graphics::internal::context;
	constexpr uint32_t material_count = scene::material_count;

	const VkDescriptorPoolSize sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = material_count },
		{ .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 * material_count },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + material_count,
		.poolSizeCount = 4,
		.pPoolSizes = sizes,
	};
	if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
		std::cerr << "Failed to create the descriptor pool\n";
		return false;
	}

	auto allocate = [&](VkDescriptorSetLayout layout, VkDescriptorSet& out) {
		const VkDescriptorSetAllocateInfo info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &layout,
		};
		if (vkAllocateDescriptorSets(ctx.device, &info, &out) != VK_SUCCESS) {
			std::cerr << "Failed to allocate a descriptor set\n";
			return false;
		}
		return true;
	};

	// набор 0: сцена
	if (!allocate(scene_set_layout, scene_set) ||
	    !createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_uniforms)) {
		return false;
	}
	writeDescriptor(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_uniforms);
	// пустые массивы источников: буферы создаются сразу, чтобы дескрипторы были валидны
	uploadStorage(point_storage, 1, std::vector<GpuPointLight>());
	uploadStorage(spot_storage, 2, std::vector<GpuSpotLight>());

	// лаба 3: динамический uniform-буфер на все пирамиды. Смещение элемента должно быть кратно
	// minUniformBufferOffsetAlignment, поэтому размер ModelData округляется вверх до этой границы
	VkPhysicalDeviceProperties properties = {};
	vkGetPhysicalDeviceProperties(ctx.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	model_stride = (sizeof(ModelData) + alignment - 1) / alignment * alignment;
	if (!createBuffer(model_stride * scene::max_objects, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_uniforms)) {
		return false;
	}

	// набор 1: по одному на материал. Во всех один и тот же динамический uniform-буфер
	// (дескриптор видит одну ModelData, какую именно — задаёт смещение при привязке), текстуры у каждого свои
	for (GpuMaterial& m : materials) {
		if (!allocate(model_set_layout, m.set)) {
			return false;
		}
		writeDescriptor(m.set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, model_uniforms, sizeof(ModelData));
		writeTextureDescriptor(m.set, 1, m.albedo);
		writeTextureDescriptor(m.set, 2, m.specular);
		writeTextureDescriptor(m.set, 3, m.emissive);
	}
	return true;
}

bool createResources() {
	const VkCommandPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = graphics::internal::context.graphics_queue_index,
	};
	if (vkCreateCommandPool(graphics::internal::context.device, &pool_info, nullptr, &upload_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create the command pool for copying\n";
		return false;
	}

	if (!createMesh(geometry::makePyramid(), pyramid_mesh) || !createMesh(geometry::makeCube(), cube_mesh) ||
	    !createMesh(geometry::makeSpotCone(), cone_mesh) || !createTextures() || !createLayouts() ||
	    !createDescriptors()) {
		return false;
	}
	lit_pipeline = createPipeline("shaders/lit.vert.spv", "shaders/lit.frag.spv", lit_layout,
	                              VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, sizeof(geometry::Vertex), 3,
	                              VK_CULL_MODE_BACK_BIT);
	cube_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", unlit_layout,
	                               VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, sizeof(geometry::Vertex), 1,
	                               VK_CULL_MODE_NONE);
	line_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", unlit_layout,
	                               VK_PRIMITIVE_TOPOLOGY_LINE_LIST, sizeof(geometry::Vertex), 1, VK_CULL_MODE_NONE);
	return lit_pipeline != VK_NULL_HANDLE && cube_pipeline != VK_NULL_HANDLE && line_pipeline != VK_NULL_HANDLE;
}

// ---------- Данные кадра ----------

void uploadFrameData(const glm::mat4& view_proj) {
	const scene::DirectionalLight& dir = world.directional;
	const SceneData scene_data = {
		.view_proj = view_proj,
		.camera_position = glm::vec4(world.camera.position,
		                             world.camera.projection == scene::Projection::Perspective ? 1.0f : 0.0f),
		.camera_forward = glm::vec4(scene::cameraForward(world.camera), 0.0f),
		.ambient = glm::vec4(world.ambient.color * world.ambient.intensity, 0.0f),
		.light_direction = glm::vec4(scene::directionFromAngles(dir.yaw_degrees, dir.pitch_degrees), 0.0f),
		.light_radiance = glm::vec4(dir.color * dir.intensity, 0.0f),
		.light_counts = glm::uvec4(world.point_lights.size(), world.spot_lights.size(), 0, 0),
	};
	upload(scene_uniforms, &scene_data, sizeof(scene_data));

	// лаба 3: свойства i-й пирамиды лежат в динамическом uniform-буфере со смещением i * model_stride
	for (int i = 0; i < world.count; ++i) {
		const scene::Object& o = world.objects[i];
		const glm::mat4 model = scene::modelMatrix(o);
		const ModelData model_data = {
			.model = model,
			.normal_matrix = glm::transpose(glm::inverse(model)),
			.specular = glm::vec4(o.material.specular, o.material.shininess),
		};
		upload(model_uniforms, &model_data, sizeof(model_data), model_stride * VkDeviceSize(i));
	}

	std::vector<GpuPointLight> points;
	for (const scene::PointLight& l : world.point_lights) {
		points.push_back({ glm::vec4(l.position, 1.0f), glm::vec4(l.color * l.intensity, 0.0f) });
	}
	uploadStorage(point_storage, 1, points);

	std::vector<GpuSpotLight> spots;
	for (const scene::SpotLight& l : world.spot_lights) {
		spots.push_back({
			.position = glm::vec4(l.position, 1.0f),
			.direction = glm::vec4(scene::directionFromAngles(l.yaw_degrees, l.pitch_degrees), 0.0f),
			.radiance = glm::vec4(l.color * l.intensity, 0.0f),
			.cutoff = glm::vec4(std::cos(glm::radians(l.inner_degrees)), std::cos(glm::radians(l.outer_degrees)),
			                    0.0f, 0.0f),
		});
	}
	uploadStorage(spot_storage, 2, spots);
}

void drawUnlit(VkCommandBuffer cmd, const GpuMesh& mesh, const glm::mat4& mvp, const glm::vec3& color) {
	const UnlitPush push = { .mvp = mvp, .color = glm::vec4(color, 1.0f) };
	vkCmdPushConstants(cmd, unlit_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
	                   sizeof(push), &push);
	drawMesh(cmd, mesh);
}

// Матрица конуса прожектора: единичный конус (ось +Z) растягивается до длины 1 и радиуса по внешнему углу,
// ось Z поворачивается вдоль направления прожектора, вершина переносится в позицию источника
glm::mat4 spotConeMatrix(const scene::SpotLight& l) {
	constexpr float length = 1.0f;
	const glm::vec3 d = scene::directionFromAngles(l.yaw_degrees, l.pitch_degrees);
	// два единичных вектора, перпендикулярных оси
	const glm::vec3 helper = std::abs(d.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
	const glm::vec3 u = glm::normalize(glm::cross(d, helper));
	const glm::vec3 v = glm::cross(d, u);
	const glm::mat4 basis(glm::vec4(u, 0.0f), glm::vec4(v, 0.0f), glm::vec4(d, 0.0f), glm::vec4(0, 0, 0, 1));
	const float radius = length * std::tan(glm::radians(l.outer_degrees));
	return xform::translation(l.position) * basis * xform::scaling(glm::vec3(radius, radius, length));
}

} // namespace

bool initialize() {
	if (!createResources()) {
		shutdown();
		return false;
	}
	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkDeviceWaitIdle(ctx.device);

	destroyBuffer(scene_uniforms);
	destroyBuffer(model_uniforms);
	destroyBuffer(point_storage);
	destroyBuffer(spot_storage);
	for (GpuMesh* mesh : { &pyramid_mesh, &cube_mesh, &cone_mesh }) {
		destroyBuffer(mesh->vertices);
		destroyBuffer(mesh->indices);
	}
	for (GpuMaterial& m : materials) {
		destroyTexture(m.albedo);
		destroyTexture(m.specular);
		destroyTexture(m.emissive);
	}
	destroyTexture(black_texture);
	vkDestroySampler(ctx.device, sampler, nullptr);
	vkDestroyCommandPool(ctx.device, upload_pool, nullptr);

	vkDestroyDescriptorPool(ctx.device, pool, nullptr);
	vkDestroyPipeline(ctx.device, lit_pipeline, nullptr);
	vkDestroyPipeline(ctx.device, cube_pipeline, nullptr);
	vkDestroyPipeline(ctx.device, line_pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, lit_layout, nullptr);
	vkDestroyPipelineLayout(ctx.device, unlit_layout, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, model_set_layout, nullptr);
}

void update(double time) {
	const float dt = previous_time < 0.0 ? 0.0f : float(std::min(time - previous_time, 0.1));
	previous_time = time;
	current_time = float(time);

	for (int i = 0; i < world.count; ++i) {
		scene::step(world.objects[i].motion, dt);
	}
	ui::drawSceneWindow(world);
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& ctx = graphics::internal::context;
	const VkCommandBuffer cmd = fd.command_buffer;
	const VkExtent2D extent = ctx.swapchain_extent;
	const float aspect = float(extent.width) / float(std::max(extent.height, 1u));
	const glm::mat4 view_proj = scene::projectionMatrix(world.camera, aspect) * scene::viewMatrix(world.camera);

	uploadFrameData(view_proj);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin);

	const VkClearValue clears[] = {
		{ .color = { .float32 = { 0.02f, 0.02f, 0.03f, 1.0f } } },
		{ .depthStencil = { .depth = 1.0f, .stencil = 0 } },
	};
	const VkRenderPassBeginInfo pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = ctx.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = extent },
		.clearValueCount = 2,
		.pClearValues = clears,
	};
	vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = { .width = float(extent.width), .height = float(extent.height), .maxDepth = 1.0f };
	const VkRect2D scissor = { .extent = extent };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	// пирамиды с освещением: набор 0 (сцена) привязывается один раз,
	// набор 1 — набор материала пирамиды со смещением её данных в динамическом uniform-буфере
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_layout, 0, 1, &scene_set, 0, nullptr);
	for (int i = 0; i < world.count; ++i) {
		const scene::Object& o = world.objects[i];
		const uint32_t offset = uint32_t(model_stride * VkDeviceSize(i));
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_layout, 1, 1,
		                        &materials[o.material.textures].set, 1, &offset);

		const LitPush push = { .time = current_time, .flags = o.animated_sampling ? flag_animated_sampling : 0u };
		vkCmdPushConstants(cmd, lit_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
		drawMesh(cmd, pyramid_mesh);
	}

	// визуализация источников: постоянный цвет, без освещения
	if (world.show_lights) {
		const glm::mat4 cube_scale = xform::scaling(glm::vec3(0.1f));
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cube_pipeline);
		for (const scene::PointLight& l : world.point_lights) {
			drawUnlit(cmd, cube_mesh, view_proj * xform::translation(l.position) * cube_scale, l.color);
		}
		for (const scene::SpotLight& l : world.spot_lights) {
			drawUnlit(cmd, cube_mesh, view_proj * xform::translation(l.position) * cube_scale, l.color);
		}

		// конусы прожекторов белыми линиями: ось, 8 образующих и окружность основания (по внешнему углу)
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
		for (const scene::SpotLight& l : world.spot_lights) {
			drawUnlit(cmd, cone_mesh, view_proj * spotConeMatrix(l), glm::vec3(1.0f));
		}
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
