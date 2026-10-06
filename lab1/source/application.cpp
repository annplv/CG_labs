#include "application.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>

#include "geometry.hpp"
#include "scene.hpp"
#include "ui.hpp"

namespace application {

namespace {

struct ObjectData {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 tint;
	glm::vec4 params;
};
static_assert(sizeof(ObjectData) == 224);

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
};

struct ObjectSlot {
	Buffer uniforms;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

Buffer vertex_buffer;
Buffer index_buffer;
uint32_t index_count = 0;

VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkDescriptorPool pool = VK_NULL_HANDLE;
std::array<ObjectSlot, scene::max_objects> slots;

scene::Scene world;
double previous_time = -1.0;

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
	return true;
}

void upload(const Buffer& buffer, const void* data, size_t size) {
	std::memcpy(buffer.mapped, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
	}
	buffer = {};
}

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

bool createPipeline() {
	auto& ctx = graphics::internal::context;

	const VkShaderModule vert = loadShader("shaders/mesh.vert.spv");
	const VkShaderModule frag = loadShader("shaders/mesh.frag.spv");
	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
		vkDestroyShaderModule(ctx.device, vert, nullptr);
		vkDestroyShaderModule(ctx.device, frag, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
		  .module = vert, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		  .module = frag, .pName = "main" },
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0, .stride = sizeof(geometry::Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, position)) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(geometry::Vertex, color)) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo raster = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
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
		.layout = pipeline_layout,
		.renderPass = ctx.render_pass,
		.subpass = 0,
	};
	const VkResult result = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);

	vkDestroyShaderModule(ctx.device, vert, nullptr);
	vkDestroyShaderModule(ctx.device, frag, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create the graphics pipeline\n";
		pipeline = VK_NULL_HANDLE;
		return false;
	}
	return true;
}

bool createResources() {
	auto& ctx = graphics::internal::context;

	const geometry::Mesh mesh = geometry::makePyramid();
	index_count = uint32_t(mesh.indices.size());
	const size_t vertices_size = mesh.vertices.size() * sizeof(geometry::Vertex);
	const size_t indices_size = mesh.indices.size() * sizeof(uint16_t);

	if (!createBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
		return false;
	}
	upload(vertex_buffer, mesh.vertices.data(), vertices_size);
	upload(index_buffer, mesh.indices.data(), indices_size);

	const VkDescriptorSetLayoutBinding layout_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &layout_binding,
	};
	if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create the descriptor set layout\n";
		return false;
	}

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &set_layout,
	};
	if (vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create the pipeline layout\n";
		return false;
	}

	if (!createPipeline()) {
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = scene::max_objects,
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = scene::max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
		std::cerr << "Failed to create the descriptor pool\n";
		return false;
	}

	for (ObjectSlot& slot : slots) {
		if (!createBuffer(sizeof(ObjectData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, slot.uniforms)) {
			return false;
		}

		const VkDescriptorSetAllocateInfo allocate = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &set_layout,
		};
		if (vkAllocateDescriptorSets(ctx.device, &allocate, &slot.set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate a descriptor set\n";
			return false;
		}

		const VkDescriptorBufferInfo buffer_info = { .buffer = slot.uniforms.handle, .offset = 0,
		                                             .range = sizeof(ObjectData) };
		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = slot.set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};
		vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
	}
	return true;
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

	for (ObjectSlot& slot : slots) {
		destroyBuffer(slot.uniforms);
	}
	vkDestroyDescriptorPool(ctx.device, pool, nullptr);
	vkDestroyPipeline(ctx.device, pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, set_layout, nullptr);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);
	pool = VK_NULL_HANDLE;
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	set_layout = VK_NULL_HANDLE;
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
	const VkExtent2D extent = ctx.swapchain_extent;
	const float aspect = float(extent.width) / float(std::max(extent.height, 1u));

	const glm::mat4 view = scene::viewMatrix(world.camera);
	const glm::mat4 proj = scene::projectionMatrix(world.camera, aspect);

	for (int i = 0; i < world.count; ++i) {
		const scene::Object& object = world.objects[i];
		const ObjectData data = {
			.model = scene::modelMatrix(object),
			.view = view,
			.proj = proj,
			.tint = glm::vec4(object.color, 1.0f), 
			.params = glm::vec4(object.vertex_colors ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f),
		};
		upload(slots[i].uniforms, &data, sizeof(data));
	}

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin);

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
	vkCmdBeginRenderPass(fd.command_buffer, &pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = { .width = float(extent.width), .height = float(extent.height), .maxDepth = 1.0f };
	const VkRect2D scissor = { .extent = extent };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);

	// у каждой пирамиды свой descriptor set (матрицы и цвет)
	for (int i = 0; i < world.count; ++i) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
		                        &slots[i].set, 0, nullptr);
		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
