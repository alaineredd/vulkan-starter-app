#include "application.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

#include <GLFW/glfw3.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <cstddef>

namespace {

    constexpr uint32_t MAX_LIGHTS = 32;

    struct Vertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec3 color;
    };

    // Данные сцены: камера и направленный свет.
    struct alignas(16) SceneUniformBufferObject {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
        alignas(16) glm::vec4 cameraPos;
        alignas(16) glm::vec4 dirDirection;
        alignas(16) glm::vec4 dirColorIntensity;
        alignas(16) glm::vec4 ambient;
        alignas(16) glm::ivec4 counts;
    };

    // Данные отдельной модели: трансформация и материал.
    struct alignas(16) ModelUniformBufferObject {
        alignas(16) glm::mat4 model;
        alignas(16) glm::vec4 albedo;
        alignas(16) glm::vec4 specular;
        alignas(16) glm::vec4 material;
    };

    struct alignas(16) PointLight {
        alignas(16) glm::vec4 positionIntensity;
        alignas(16) glm::vec4 color;
    };

    struct alignas(16) SpotLight {
        alignas(16) glm::vec4 positionIntensity;
        alignas(16) glm::vec4 directionInner;
        alignas(16) glm::vec4 colorOuter;
    };

    SceneUniformBufferObject scene_ubo{};
    ModelUniformBufferObject model_ubo{};

    VkPipeline graphics_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout scene_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout model_descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
    VkDescriptorSet model_descriptor_set = VK_NULL_HANDLE;

    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;
    uint32_t index_count = 0;

    VkBuffer scene_uniform_buffer = VK_NULL_HANDLE;
    VmaAllocation scene_uniform_buffer_allocation = VK_NULL_HANDLE;
    void* scene_uniform_buffer_mapped = nullptr;
    VkBuffer model_uniform_buffer = VK_NULL_HANDLE;
    VmaAllocation model_uniform_buffer_allocation = VK_NULL_HANDLE;
    void* model_uniform_buffer_mapped = nullptr;

    VkBuffer point_lights_buffer = VK_NULL_HANDLE;
    VmaAllocation point_lights_buffer_allocation = VK_NULL_HANDLE;
    void* point_lights_buffer_mapped = nullptr;
    VkBuffer spot_lights_buffer = VK_NULL_HANDLE;
    VmaAllocation spot_lights_buffer_allocation = VK_NULL_HANDLE;
    void* spot_lights_buffer_mapped = nullptr;

    bool use_perspective = true;
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 rotation = glm::vec3(0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    // аним
    bool animation_enabled = false;
    float animation_speed = 1.0f;
    float animation_radius = 2.0f;
    float animation_time = 0.0f;

    // Параметры камеры.
    glm::vec3 camera_position = glm::vec3(0.0f, 0.0f, 5.0f);
    glm::vec3 camera_rotation = glm::vec3(0.0f);
    float camera_fov = 45.0f;

    // Параметры материала и освещения.
    glm::vec3 albedo = glm::vec3(0.8f, 0.35f, 0.18f);
    glm::vec3 specular_color = glm::vec3(1.0f);
    float shininess = 32.0f;
    float ambient_strength = 0.12f;
    bool directional_enabled = true;
    glm::vec3 directional_direction = glm::vec3(-0.5f, -1.0f, -0.5f);
    glm::vec3 directional_color = glm::vec3(1.0f);
    float directional_intensity = 1.0f;

    std::vector<PointLight> point_lights = {
        { glm::vec4(2.0f, 1.0f, 2.0f, 18.0f), glm::vec4(1.0f, 0.25f, 0.15f, 1.0f) }
    };
    std::vector<SpotLight> spot_lights = {
        { glm::vec4(-2.0f, 2.0f, 2.0f, 24.0f),
          glm::vec4(0.0f, -0.6f, -1.0f, std::cos(glm::radians(15.0f))),
          glm::vec4(0.2f, 0.45f, 1.0f, std::cos(glm::radians(23.0f))) }
    };

    double last_time = 0.0;

    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Working directory: "
                << std::filesystem::current_path().string()
                << std::endl;
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
            // Нормаль генерируется процедурно из позиции вершины.
            vertices[i].normal = glm::normalize(positions[i]);
            // Процедурные цвета из лабораторной №1 сохраняются.

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
        // UBO сцены, UBO модели и storage-буферы для массивов источников.
        createBuffer(sizeof(SceneUniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, scene_uniform_buffer, scene_uniform_buffer_allocation,
            &scene_uniform_buffer_mapped);
        createBuffer(sizeof(ModelUniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, model_uniform_buffer, model_uniform_buffer_allocation,
            &model_uniform_buffer_mapped);
        createBuffer(sizeof(PointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, point_lights_buffer, point_lights_buffer_allocation,
            &point_lights_buffer_mapped);
        createBuffer(sizeof(SpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, spot_lights_buffer, spot_lights_buffer_allocation,
            &spot_lights_buffer_mapped);

        // Набор №0: данные сцены и storage-буферы источников освещения.
        std::array<VkDescriptorSetLayoutBinding, 3> scene_bindings{};
        scene_bindings[0].binding = 0;
        scene_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        scene_bindings[0].descriptorCount = 1;
        scene_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        scene_bindings[1].binding = 1;
        scene_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        scene_bindings[1].descriptorCount = 1;
        scene_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        scene_bindings[2].binding = 2;
        scene_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        scene_bindings[2].descriptorCount = 1;
        scene_bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo scene_layout_info{};
        scene_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        scene_layout_info.bindingCount = static_cast<uint32_t>(scene_bindings.size());
        scene_layout_info.pBindings = scene_bindings.data();
        if (vkCreateDescriptorSetLayout(ctx.device, &scene_layout_info, nullptr,
            &scene_descriptor_set_layout) != VK_SUCCESS) return false;

        // Набор №1: матрица модели и параметры материала.
        VkDescriptorSetLayoutBinding model_binding{};
        model_binding.binding = 0;
        model_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        model_binding.descriptorCount = 1;
        model_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo model_layout_info{};
        model_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        model_layout_info.bindingCount = 1;
        model_layout_info.pBindings = &model_binding;
        if (vkCreateDescriptorSetLayout(ctx.device, &model_layout_info, nullptr,
            &model_descriptor_set_layout) != VK_SUCCESS) return false;

        std::array<VkDescriptorPoolSize, 2> pool_sizes{};
        pool_sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 };
        pool_sizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.maxSets = 2;
        if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) return false;

        std::array<VkDescriptorSetLayout, 2> set_layouts = {
            scene_descriptor_set_layout, model_descriptor_set_layout
        };
        std::array<VkDescriptorSet, 2> descriptor_sets{};
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = descriptor_pool;
        alloc_info.descriptorSetCount = static_cast<uint32_t>(set_layouts.size());
        alloc_info.pSetLayouts = set_layouts.data();
        if (vkAllocateDescriptorSets(ctx.device, &alloc_info, descriptor_sets.data()) != VK_SUCCESS) return false;
        scene_descriptor_set = descriptor_sets[0];
        model_descriptor_set = descriptor_sets[1];

        std::array<VkDescriptorBufferInfo, 4> buffer_infos{};
        buffer_infos[0] = { scene_uniform_buffer, 0, sizeof(SceneUniformBufferObject) };
        buffer_infos[1] = { point_lights_buffer, 0, sizeof(PointLight) * MAX_LIGHTS };
        buffer_infos[2] = { spot_lights_buffer, 0, sizeof(SpotLight) * MAX_LIGHTS };
        buffer_infos[3] = { model_uniform_buffer, 0, sizeof(ModelUniformBufferObject) };
        std::array<VkWriteDescriptorSet, 4> descriptor_writes{};
        for (uint32_t i = 0; i < 3; ++i) {
            descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[i].dstSet = scene_descriptor_set;
            descriptor_writes[i].dstBinding = i;
            descriptor_writes[i].descriptorCount = 1;
            descriptor_writes[i].descriptorType = (i == 0)
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptor_writes[i].pBufferInfo = &buffer_infos[i];
        }
        descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[3].dstSet = model_descriptor_set;
        descriptor_writes[3].dstBinding = 0;
        descriptor_writes[3].descriptorCount = 1;
        descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_writes[3].pBufferInfo = &buffer_infos[3];
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(descriptor_writes.size()),
            descriptor_writes.data(), 0, nullptr);

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        std::array<VkDescriptorSetLayout, 2> pipeline_set_layouts = {
            scene_descriptor_set_layout, model_descriptor_set_layout
        };
        pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(pipeline_set_layouts.size());
        pipeline_layout_info.pSetLayouts = pipeline_set_layouts.data();
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

        std::array<VkVertexInputAttributeDescription, 3> attribute_descs{};
        attribute_descs[0].binding = 0;
        attribute_descs[0].location = 0;
        attribute_descs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descs[0].offset = offsetof(Vertex, pos);

        attribute_descs[1].binding = 0;
        attribute_descs[1].location = 1;
        attribute_descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descs[1].offset = offsetof(Vertex, normal);

        attribute_descs[2].binding = 0;
        attribute_descs[2].location = 2;
        attribute_descs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descs[2].offset = offsetof(Vertex, color);

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
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
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

        if (graphics_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, graphics_pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        if (scene_descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, scene_descriptor_set_layout, nullptr);
        if (model_descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, model_descriptor_set_layout, nullptr);

        if (scene_uniform_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, scene_uniform_buffer, scene_uniform_buffer_allocation);
        if (model_uniform_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, model_uniform_buffer, model_uniform_buffer_allocation);
        if (point_lights_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, point_lights_buffer, point_lights_buffer_allocation);
        if (spot_lights_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, spot_lights_buffer, spot_lights_buffer_allocation);
        if (index_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, index_buffer, index_buffer_allocation);
        if (vertex_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_buffer_allocation);
    }

    void update(double time) {
        // Вычисление времени между кадрами.
        double delta = std::clamp(time - last_time, 0.0, 0.1);
        last_time = time;

        if (animation_enabled) {
            animation_time += static_cast<float>(delta) * animation_speed;
        }

        ImGui::Begin("Lab 2 - Camera and Lighting");

        if (ImGui::CollapsingHeader("Model and material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Object position", &position.x, -5.0f, 5.0f);
            ImGui::SliderFloat3("Object rotation", &rotation.x, -180.0f, 180.0f);
            ImGui::SliderFloat3("Object scale", &scale.x, 0.1f, 3.0f);

            ImGui::Separator();
            ImGui::Checkbox("Animation", &animation_enabled);

            if (animation_enabled) {
                ImGui::SliderFloat("Animation Speed", &animation_speed, 0.0f, 5.0f);
                ImGui::SliderFloat("Animation Radius", &animation_radius, 0.0f, 5.0f);
            }

            ImGui::Separator();
            ImGui::ColorEdit3("Albedo", &albedo.x);
            ImGui::ColorEdit3("Specular color", &specular_color.x);
            ImGui::SliderFloat("Shininess", &shininess, 1.0f, 128.0f);
        }

        if (ImGui::CollapsingHeader("Model and material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Object position", &position.x, -5.0f, 5.0f);
            ImGui::SliderFloat3("Object rotation", &rotation.x, -180.0f, 180.0f);
            ImGui::SliderFloat3("Object scale", &scale.x, 0.1f, 3.0f);
            ImGui::ColorEdit3("Albedo", &albedo.x);
            ImGui::ColorEdit3("Specular color", &specular_color.x);
            ImGui::SliderFloat("Shininess", &shininess, 1.0f, 128.0f);
        }

        if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable directional light", &directional_enabled);
            ImGui::SliderFloat3("Direction", &directional_direction.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &directional_color.x);
            ImGui::SliderFloat("Intensity", &directional_intensity, 0.0f, 5.0f);
            ImGui::SliderFloat("Ambient light", &ambient_strength, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (point_lights.size() < MAX_LIGHTS && ImGui::Button("Add point light")) {
                point_lights.push_back({ glm::vec4(1.0f, 1.0f, 2.0f, 15.0f), glm::vec4(1.0f, 0.8f, 0.5f, 1.0f) });
            }
            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                std::string label = "Point " + std::to_string(i);
                if (ImGui::TreeNode(label.c_str())) {
                    PointLight& light = point_lights[i];
                    ImGui::SliderFloat3("Position", &light.positionIntensity.x, -5.0f, 5.0f);
                    ImGui::SliderFloat("Intensity", &light.positionIntensity.w, 0.0f, 100.0f);
                    ImGui::ColorEdit3("Color", &light.color.x);
                    if (ImGui::Button("Remove")) {
                        point_lights.erase(point_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (spot_lights.size() < MAX_LIGHTS && ImGui::Button("Add spot light")) {
                spot_lights.push_back({ glm::vec4(0.0f, 2.0f, 2.0f, 25.0f),
                    glm::vec4(0.0f, -1.0f, -1.0f, std::cos(glm::radians(15.0f))),
                    glm::vec4(1.0f, 1.0f, 0.8f, std::cos(glm::radians(25.0f))) });
            }
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID(1000 + static_cast<int>(i));
                std::string label = "Spot " + std::to_string(i);
                if (ImGui::TreeNode(label.c_str())) {
                    SpotLight& light = spot_lights[i];
                    ImGui::SliderFloat3("Position", &light.positionIntensity.x, -5.0f, 5.0f);
                    ImGui::SliderFloat("Intensity", &light.positionIntensity.w, 0.0f, 100.0f);
                    ImGui::SliderFloat3("Direction", &light.directionInner.x, -1.0f, 1.0f);
                    float inner = glm::degrees(std::acos(glm::clamp(light.directionInner.w, -1.0f, 1.0f)));
                    float outer = glm::degrees(std::acos(glm::clamp(light.colorOuter.w, -1.0f, 1.0f)));
                    if (ImGui::SliderFloat("Inner angle", &inner, 1.0f, 60.0f))
                        light.directionInner.w = std::cos(glm::radians(inner));
                    if (ImGui::SliderFloat("Outer angle", &outer, inner, 75.0f))
                        light.colorOuter.w = std::cos(glm::radians(std::max(outer, inner)));
                    if (outer < inner) light.colorOuter.w = std::cos(glm::radians(inner));
                    ImGui::ColorEdit3("Color", &light.colorOuter.x);
                    if (ImGui::Button("Remove")) {
                        spot_lights.erase(spot_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }
        ImGui::End();

        // Матрица модели сохранена из лабораторной №1.
        glm::mat4 model = glm::mat4(1.0f);

        // Текущая позиция объекта.
        glm::vec3 current_pos = position;

        if (animation_enabled) {
            current_pos.x += animation_radius * std::cos(animation_time);
            current_pos.y += animation_radius * std::sin(animation_time);
        }

        model = glm::translate(model, current_pos);

        // Текущий поворот объекта.
        float rot_x = rotation.x;
        float rot_y = rotation.y;
        float rot_z = rotation.z;

        if (animation_enabled) {
            rot_z += glm::degrees(animation_time * 50.0f);
        }

        model = glm::rotate(
            model,
            glm::radians(rot_x),
            glm::vec3(1.0f, 0.0f, 0.0f)
        );

        model = glm::rotate(
            model,
            glm::radians(rot_y),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        model = glm::rotate(
            model,
            glm::radians(rot_z),
            glm::vec3(0.0f, 0.0f, 1.0f)
        );

        model = glm::scale(model, scale);

        glm::vec3 forward;
        forward.x = std::cos(glm::radians(camera_rotation.x)) * std::sin(glm::radians(camera_rotation.y));
        forward.y = std::sin(glm::radians(camera_rotation.x));
        forward.z = -std::cos(glm::radians(camera_rotation.x)) * std::cos(glm::radians(camera_rotation.y));
        glm::mat4 view = glm::lookAt(camera_position, camera_position + forward, glm::vec3(0.0f, 1.0f, 0.0f));

        const auto& extent = graphics::internal::context.swapchain_extent;
        float aspect = static_cast<float>(extent.width) / static_cast<float>(std::max(1u, extent.height));
        glm::mat4 proj;
        if (use_perspective) {
            proj = glm::perspectiveRH_ZO(glm::radians(camera_fov), aspect, 0.1f, 100.0f);
        }
        else {
            proj = glm::orthoRH_ZO(-3.0f * aspect, 3.0f * aspect, -3.0f, 3.0f, 0.1f, 100.0f);
        }
        proj[1][1] *= -1.0f;

        scene_ubo.view = view;
        scene_ubo.proj = proj;
        scene_ubo.cameraPos = glm::vec4(camera_position, 1.0f);
        scene_ubo.dirDirection = glm::vec4(glm::normalize(directional_direction), 0.0f);
        scene_ubo.dirColorIntensity = glm::vec4(directional_color, directional_intensity);
        scene_ubo.ambient = glm::vec4(glm::vec3(ambient_strength), 1.0f);
        scene_ubo.counts = glm::ivec4(static_cast<int>(point_lights.size()), static_cast<int>(spot_lights.size()), directional_enabled ? 1 : 0, 0);

        model_ubo.model = model;
        model_ubo.albedo = glm::vec4(albedo, 1.0f);
        model_ubo.specular = glm::vec4(specular_color, 1.0f);
        model_ubo.material = glm::vec4(shininess, 0.0f, 0.0f, 0.0f);

        std::memcpy(scene_uniform_buffer_mapped, &scene_ubo, sizeof(scene_ubo));
        std::memcpy(model_uniform_buffer_mapped, &model_ubo, sizeof(model_ubo));
        if (!point_lights.empty()) std::memcpy(point_lights_buffer_mapped, point_lights.data(), sizeof(PointLight) * point_lights.size());
        if (!spot_lights.empty()) std::memcpy(spot_lights_buffer_mapped, spot_lights.data(), sizeof(SpotLight) * spot_lights.size());
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;
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

        VkDescriptorSet descriptor_sets[] = { scene_descriptor_set, model_descriptor_set };
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
            0, 2, descriptor_sets, 0, nullptr);

        VkBuffer vertex_buffers[] = { vertex_buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, vertex_buffers, offsets);

        vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(fd.command_buffer);

        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application