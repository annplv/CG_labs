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

#include "geometry.hpp"
#include "scene.hpp"
#include "transform.hpp"
#include "ui.hpp"

namespace application {

namespace {

// ---------- Данные для шейдеров (раскладка совпадает с GLSL) ----------

// набор 0, binding 0 (std140)
struct SceneData {
	glm::mat4 view_proj;
	glm::vec4 camera_position;
	glm::vec4 camera_forward;
	glm::vec4 ambient;
	glm::vec4 light_direction;
	glm::vec4 light_radiance;
	glm::uvec4 light_counts;
};

// набор 0, binding 1 (std430)
struct GpuPointLight {
	glm::vec4 position;
	glm::vec4 radiance;
};

// набор 0, binding 2 (std430)
struct GpuSpotLight {
	glm::vec4 position;
	glm::vec4 direction;
	glm::vec4 radiance;
	glm::vec4 cutoff;
};

// набор 1, binding 0 (std140)
struct ModelData {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 albedo;
	glm::vec4 specular;
};

// push-константы для визуализации источников
struct UnlitPush {
	glm::mat4 mvp;
	glm::vec4 color;
};

// ---------- Ресурсы ----------

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

GpuMesh pyramid_mesh;
GpuMesh cube_mesh;

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

// как в лабе 1: у каждой пирамиды свой буфер и свой набор дескрипторов (теперь это набор 1 — модель)
struct ObjectSlot {
	Buffer uniforms;
	VkDescriptorSet set = VK_NULL_HANDLE;
};
std::array<ObjectSlot, scene::max_objects> slots;

Buffer line_vertices;

scene::Scene world;
double previous_time = -1.0;

// ---------- Буферы ----------

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

void upload(const Buffer& buffer, const void* data, size_t size) {
	if (size == 0) {
		return;
	}
	std::memcpy(buffer.mapped, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
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

void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buffer) {
	const VkDescriptorBufferInfo info = { .buffer = buffer.handle, .offset = 0, .range = VK_WHOLE_SIZE };
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

bool createMesh(const geometry::Mesh& mesh, GpuMesh& out) {
	const size_t vertices_size = mesh.vertices.size() * sizeof(geometry::Vertex);
	const size_t indices_size = mesh.indices.size() * sizeof(uint16_t);
	if (!createBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, out.vertices) ||
	    !createBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, out.indices)) {
		return false;
	}
	upload(out.vertices, mesh.vertices.data(), vertices_size);
	upload(out.indices, mesh.indices.data(), indices_size);
	out.index_count = uint32_t(mesh.indices.size());
	return true;
}

void drawMesh(VkCommandBuffer cmd, const GpuMesh& mesh) {
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertices.handle, &offset);
	vkCmdBindIndexBuffer(cmd, mesh.indices.handle, 0, VK_INDEX_TYPE_UINT16);
	vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
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
	// location 0 — позиция, 1 — нормаль, 2 — цвет вершины (нормаль и цвет нужны только освещённому пайплайну)
	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, position)) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, normal)) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, color)) },
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

	// набор 1 (модель): матрица трансформации и материал
	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = both,
	};
	const VkDescriptorSetLayoutCreateInfo model_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(ctx.device, &scene_info, nullptr, &scene_set_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(ctx.device, &model_info, nullptr, &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	const VkDescriptorSetLayout lit_sets[] = { scene_set_layout, model_set_layout };
	const VkPipelineLayoutCreateInfo lit_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = lit_sets,
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

	const VkDescriptorPoolSize sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + scene::max_objects },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + scene::max_objects,
		.poolSizeCount = 2,
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

	// набор 1: по одному на каждую пирамиду
	for (ObjectSlot& slot : slots) {
		if (!allocate(model_set_layout, slot.set) ||
		    !createBuffer(sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, slot.uniforms)) {
			return false;
		}
		writeDescriptor(slot.set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, slot.uniforms);
	}
	return true;
}

bool createResources() {
	if (!createMesh(geometry::makePyramid(), pyramid_mesh) || !createMesh(geometry::makeCube(), cube_mesh) ||
	    !createLayouts() || !createDescriptors()) {
		return false;
	}
	lit_pipeline = createPipeline("shaders/lit.vert.spv", "shaders/lit.frag.spv", lit_layout,
	                              VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, sizeof(geometry::Vertex), 3,
	                              VK_CULL_MODE_BACK_BIT);
	cube_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", unlit_layout,
	                               VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, sizeof(geometry::Vertex), 1,
	                               VK_CULL_MODE_NONE);
	line_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", unlit_layout,
	                               VK_PRIMITIVE_TOPOLOGY_LINE_LIST, sizeof(glm::vec3), 1, VK_CULL_MODE_NONE);
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

	for (int i = 0; i < world.count; ++i) {
		const scene::Object& o = world.objects[i];
		const glm::mat4 model = scene::modelMatrix(o);
		const ModelData model_data = {
			.model = model,
			.normal_matrix = glm::transpose(glm::inverse(model)),
			.albedo = glm::vec4(o.material.albedo, o.vertex_colors ? 1.0f : 0.0f),
			.specular = glm::vec4(o.material.specular, o.material.shininess),
		};
		upload(slots[i].uniforms, &model_data, sizeof(model_data));
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

// Конус прожектора белыми линиями: ось, 8 образующих и окружность основания (по внешнему углу)
std::vector<glm::vec3> buildSpotCones() {
	constexpr float length = 1.0f; // длина конуса
	constexpr int segments = 32;
	std::vector<glm::vec3> lines;

	for (const scene::SpotLight& l : world.spot_lights) {
		const glm::vec3 apex = l.position;
		const glm::vec3 d = scene::directionFromAngles(l.yaw_degrees, l.pitch_degrees);
		// два единичных вектора, перпендикулярных оси
		const glm::vec3 helper = std::abs(d.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
		const glm::vec3 u = glm::normalize(glm::cross(d, helper));
		const glm::vec3 v = glm::cross(d, u);
		const float radius = length * std::tan(glm::radians(l.outer_degrees));

		auto circle = [&](int k) {
			const float a = 2.0f * 3.14159265f * float(k) / float(segments);
			return apex + d * length + (u * std::cos(a) + v * std::sin(a)) * radius;
		};

		lines.push_back(apex);
		lines.push_back(apex + d * length); // ось
		for (int k = 0; k < segments; ++k) {
			lines.push_back(circle(k));
			lines.push_back(circle(k + 1));
			if (k % 4 == 0) { // образующая
				lines.push_back(apex);
				lines.push_back(circle(k));
			}
		}
	}
	return lines;
}

void drawLightCube(VkCommandBuffer cmd, const glm::mat4& view_proj, const glm::vec3& position,
                   const glm::vec3& color) {
	const UnlitPush push = {
		.mvp = view_proj * xform::translation(position) * xform::scaling(glm::vec3(0.1f)),
		.color = glm::vec4(color, 1.0f),
	};
	vkCmdPushConstants(cmd, unlit_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
	                   sizeof(push), &push);
	drawMesh(cmd, cube_mesh);
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
	for (ObjectSlot& slot : slots) {
		destroyBuffer(slot.uniforms);
	}
	destroyBuffer(point_storage);
	destroyBuffer(spot_storage);
	destroyBuffer(line_vertices);
	destroyBuffer(pyramid_mesh.vertices);
	destroyBuffer(pyramid_mesh.indices);
	destroyBuffer(cube_mesh.vertices);
	destroyBuffer(cube_mesh.indices);

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

	std::vector<glm::vec3> cone_lines;
	if (world.show_lights) {
		cone_lines = buildSpotCones();
		growIfNeeded(line_vertices, cone_lines.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		upload(line_vertices, cone_lines.data(), cone_lines.size() * sizeof(glm::vec3));
	}

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

	// пирамиды с освещением: набор 0 (сцена) привязывается один раз, набор 1 (модель) — для каждой пирамиды
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_layout, 0, 1, &scene_set, 0, nullptr);
	for (int i = 0; i < world.count; ++i) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_layout, 1, 1, &slots[i].set, 0, nullptr);
		drawMesh(cmd, pyramid_mesh);
	}

	// визуализация источников: постоянный цвет, без освещения
	if (world.show_lights) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cube_pipeline);
		for (const scene::PointLight& l : world.point_lights) {
			drawLightCube(cmd, view_proj, l.position, l.color);
		}
		for (const scene::SpotLight& l : world.spot_lights) {
			drawLightCube(cmd, view_proj, l.position, l.color);
		}

		if (!cone_lines.empty()) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipeline);
			const VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &line_vertices.handle, &offset);
			const UnlitPush push = { .mvp = view_proj, .color = glm::vec4(1.0f) }; // белые линии
			vkCmdPushConstants(cmd, unlit_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
			                   sizeof(push), &push);
			vkCmdDraw(cmd, uint32_t(cone_lines.size()), 1, 0, 0);
		}
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
