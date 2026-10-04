#include "application.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <cstring>
#include <cmath>

namespace {

    struct Vertex {
        glm::vec3 pos;
        glm::vec3 color;
    };

    struct UniformBufferObject {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
        alignas(16) glm::vec4 colorMultiplier;
    };

    UniformBufferObject current_ubo{};

    VkPipeline graphics_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;
    uint32_t index_count = 0;

    VkBuffer uniform_buffer = VK_NULL_HANDLE;
    VmaAllocation uniform_buffer_allocation = VK_NULL_HANDLE;
    void* uniform_buffer_mapped = nullptr;

    bool use_perspective = true;
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 rotation = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
    glm::vec3 color_multiplier = glm::vec3(1.0f);
    bool animation_enabled = false;
    float animation_speed = 1.0f;
    float animation_radius = 2.0f;
    float animation_time = 0.0f;
    double last_time = 0.0;

    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file: " + filename);
        }
        size_t file_size = (size_t)file.tellg();
        std::vector<char> buffer(file_size);
        file.seekg(0);
        file.read(buffer.data(), file_size);
        file.close();
        return buffer;
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage,
        VkBuffer& buffer, VmaAllocation& allocation, void** mapped_data = nullptr) {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = memory_usage;
        if (mapped_data) {
            alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        VmaAllocationInfo allocation_info{};
        if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info,
            &buffer, &allocation, &allocation_info) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer");
        }
        if (mapped_data) {
            *mapped_data = allocation_info.pMappedData;
        }
    }

    void generateIcosahedron(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
        const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
        const float max_coord = t;

        // вершины
        std::array<glm::vec3, 12> positions = { {
            {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
            { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
            { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1}
        } };

        vertices.resize(12);
        for (size_t i = 0; i < 12; ++i) {
            vertices[i].pos = positions[i];
            // процедурные цвета 5 пункт
            glm::vec3 normalized = positions[i] / max_coord;
            vertices[i].color = (normalized + 1.0f) * 0.5f;
        }

        // 20 треугольных граней с обходом против часовой стрелки
        indices = {
            0, 11, 5,  0, 5, 1,  0, 1, 7,  0, 7, 10, 0, 10, 11,
            1, 5, 9,  5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4,  3, 4, 2,  3, 2, 6,  3, 6, 8,  3, 8, 9,
            4, 9, 5,  2, 4, 11, 6, 2, 10, 8, 6, 7,  9, 8, 1
        };
    }

} // namespace

namespace application {

    bool initialize() {
        auto& ctx = graphics::internal::context;

        // геометрия
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        generateIcosahedron(vertices, indices);
        index_count = static_cast<uint32_t>(indices.size());

        VkDeviceSize vertex_buffer_size = sizeof(Vertex) * vertices.size();
        VkDeviceSize index_buffer_size = sizeof(uint32_t) * indices.size();

        void* vertex_data = nullptr;
        createBuffer(vertex_buffer_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, vertex_buffer, vertex_buffer_allocation, &vertex_data);
        std::memcpy(vertex_data, vertices.data(), vertex_buffer_size);

        void* index_data = nullptr;
        createBuffer(index_buffer_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, index_buffer, index_buffer_allocation, &index_data);
        std::memcpy(index_data, indices.data(), index_buffer_size);

        // буффер
        VkDeviceSize ubo_size = sizeof(UniformBufferObject);
        createBuffer(ubo_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, uniform_buffer, uniform_buffer_allocation,
            &uniform_buffer_mapped);

        VkDescriptorSetLayoutBinding ubo_layout_binding{};
        ubo_layout_binding.binding = 0;
        ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ubo_layout_binding.descriptorCount = 1;
        ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        ubo_layout_binding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 1;
        layout_info.pBindings = &ubo_layout_binding;
        if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_size.descriptorCount = 1;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        pool_info.maxSets = 1;
        if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &descriptor_set_layout;
        if (vkAllocateDescriptorSets(ctx.device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = uniform_buffer;
        buffer_info.offset = 0;
        buffer_info.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptor_write{};
        descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write.dstSet = descriptor_set;
        descriptor_write.dstBinding = 0;
        descriptor_write.dstArrayElement = 0;
        descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_write.descriptorCount = 1;
        descriptor_write.pBufferInfo = &buffer_info;

        vkUpdateDescriptorSets(ctx.device, 1, &descriptor_write, 0, nullptr);

        // лейаут
        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
        if (vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
            return false;
        }

        // грузим шейдер
        auto vert_code = readFile("shaders/icosahedron.vert.spv");
        auto frag_code = readFile("shaders/icosahedron.frag.spv");
        VkShaderModule vert_module, frag_module;
        VkShaderModuleCreateInfo vert_info{};
        vert_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vert_info.codeSize = vert_code.size();
        vert_info.pCode = reinterpret_cast<const uint32_t*>(vert_code.data());
        if (vkCreateShaderModule(ctx.device, &vert_info, nullptr, &vert_module) != VK_SUCCESS) {
            return false;
        }

        VkShaderModuleCreateInfo frag_info{};
        frag_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        frag_info.codeSize = frag_code.size();
        frag_info.pCode = reinterpret_cast<const uint32_t*>(frag_code.data());
        if (vkCreateShaderModule(ctx.device, &frag_info, nullptr, &frag_module) != VK_SUCCESS) {
            return false;
        }

        VkPipelineShaderStageCreateInfo vert_stage{};
        vert_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert_stage.module = vert_module;
        vert_stage.pName = "main";

        VkPipelineShaderStageCreateInfo frag_stage{};
        frag_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_stage.module = frag_module;
        frag_stage.pName = "main";

        VkPipelineShaderStageCreateInfo shader_stages[] = { vert_stage, frag_stage };

        VkVertexInputBindingDescription binding_desc{};
        binding_desc.binding = 0;
        binding_desc.stride = sizeof(Vertex);
        binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 2> attribute_descs{};
        attribute_descs[0].binding = 0;
        attribute_descs[0].location = 0;
        attribute_descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descs[0].offset = offsetof(Vertex, pos);

        attribute_descs[1].binding = 0;
        attribute_descs[1].location = 1;
        attribute_descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descs[1].offset = offsetof(Vertex, color);

        VkPipelineVertexInputStateCreateInfo vertex_input_info{};
        vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 1;
        vertex_input_info.pVertexBindingDescriptions = &binding_desc;
        vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descs.size());
        vertex_input_info.pVertexAttributeDescriptions = attribute_descs.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depth_stencil{};
        depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil.depthTestEnable = VK_TRUE;
        depth_stencil.depthWriteEnable = VK_TRUE;
        depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depth_stencil.depthBoundsTestEnable = VK_FALSE;
        depth_stencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState color_blend_attachment{};
        color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;

        VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = 2;
        dynamic_state.pDynamicStates = dynamic_states;

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = shader_stages;
        pipeline_info.pVertexInputState = &vertex_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pDepthStencilState = &depth_stencil;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamic_state;
        pipeline_info.layout = pipeline_layout;
        pipeline_info.renderPass = ctx.render_pass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
            &graphics_pipeline) != VK_SUCCESS) {
            return false;
        }

        vkDestroyShaderModule(ctx.device, frag_module, nullptr);
        vkDestroyShaderModule(ctx.device, vert_module, nullptr);

        last_time = glfwGetTime();
        return true;
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkQueueWaitIdle(ctx.graphics_queue);

        vkDestroyPipeline(ctx.device, graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, descriptor_set_layout, nullptr);

        vmaDestroyBuffer(ctx.allocator, uniform_buffer, uniform_buffer_allocation);
        vmaDestroyBuffer(ctx.allocator, index_buffer, index_buffer_allocation);
        vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_buffer_allocation);
    }

    void update(double time) {
        double delta = time - last_time;
        last_time = time;

        ImGui::Begin("Lab 1 - Icosahedron");
        ImGui::Checkbox("Perspective", &use_perspective);
        ImGui::SliderFloat3("Position", &position.x, -5.0f, 5.0f);
        ImGui::SliderFloat3("Rotation", &rotation.x, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale", &scale.x, 0.1f, 3.0f);
        ImGui::ColorEdit3("Color Multiplier", &color_multiplier.x);
        ImGui::Checkbox("Animation", &animation_enabled);
        if (animation_enabled) {
            ImGui::SliderFloat("Animation Speed", &animation_speed, 0.0f, 5.0f);
            ImGui::SliderFloat("Animation Radius", &animation_radius, 0.0f, 5.0f);
        }
        ImGui::End();

        if (animation_enabled) {
            animation_time += static_cast<float>(delta) * animation_speed;
        }

        // рассчет матриц
        glm::mat4 model = glm::mat4(1.0f);

        // поз
        glm::vec3 current_pos = position;
        if (animation_enabled) {
            current_pos.x += animation_radius * std::cos(animation_time);
            current_pos.y += animation_radius * std::sin(animation_time);
        }
        model = glm::translate(model, current_pos);

        // вращение
        float rot_x = rotation.x;
        float rot_y = rotation.y;
        float rot_z = rotation.z;
        if (animation_enabled) {
            rot_z += glm::degrees(animation_time * 50.0f);
        }

        model = glm::rotate(model, glm::radians(rot_x), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rot_y), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rot_z), glm::vec3(0.0f, 0.0f, 1.0f));
        // масштаб
        model = glm::scale(model, scale);

        // камера в 0 0 5
        glm::mat4 view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 5.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        float aspect = static_cast<float>(graphics::internal::context.swapchain_extent.width) /
            static_cast<float>(graphics::internal::context.swapchain_extent.height);

        glm::mat4 proj;
        if (use_perspective) {
            proj = glm::perspectiveRH_ZO(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        }
        else {
            float orthoSize = 3.0f;
            proj = glm::orthoRH_ZO(-orthoSize * aspect, orthoSize * aspect, -orthoSize, orthoSize, 0.1f, 20.0f);
        }

        proj[1][1] *= -1.0f;

        current_ubo.model = model;
        current_ubo.view = view;
        current_ubo.proj = proj;
        current_ubo.colorMultiplier = glm::vec4(color_multiplier, 1.0f);
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;
        if (uniform_buffer_mapped) {
            std::memcpy(uniform_buffer_mapped, &current_ubo, sizeof(current_ubo));
        }

        vkResetCommandBuffer(fd.command_buffer, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(fd.command_buffer, &begin_info);

        VkClearValue clear_values[2];
        clear_values[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clear_values[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = ctx.render_pass;
        render_pass_info.framebuffer = fd.framebuffer;
        render_pass_info.renderArea.offset = { 0, 0 };
        render_pass_info.renderArea.extent = ctx.swapchain_extent;
        render_pass_info.clearValueCount = 2;
        render_pass_info.pClearValues = clear_values;

        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(ctx.swapchain_extent.width);
        viewport.height = static_cast<float>(ctx.swapchain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = ctx.swapchain_extent;
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
            0, 1, &descriptor_set, 0, nullptr);

        VkBuffer vertex_buffers[] = { vertex_buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertex_buffers, offsets);

        vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(fd.command_buffer);

        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application