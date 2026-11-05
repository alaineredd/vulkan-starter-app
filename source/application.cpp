#include "application.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t max_lights = 32;
constexpr size_t model_count = 3;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
};

struct alignas(16) SceneUniformBufferObject {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 cameraPos;
    alignas(16) glm::vec4 dirDirection;
    alignas(16) glm::vec4 dirColorIntensity;
    alignas(16) glm::vec4 ambient;
    alignas(16) glm::ivec4 counts;
};

struct alignas(16) ModelUniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::vec4 albedo;
    alignas(16) glm::vec4 specular;
    alignas(16) glm::vec4 material;
};

struct alignas(16) SamplingSettings {
    float time;
    float uvScale;
    int32_t enabled;
    float blendFactor;
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

struct ModelInstance {
    const char* name;
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::vec3 albedo;
    float shininess;
    size_t textureIndex;
    bool animatedSampling;
    float uvScale;
    float blendFactor;
};

struct Texture {
    std::string name;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

SceneUniformBufferObject scene_ubo{};
std::array<ModelUniformBufferObject, model_count> model_ubos{};

VkPipeline graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout scene_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout material_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
std::vector<VkDescriptorSet> material_descriptor_sets;

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
VkDeviceSize model_uniform_stride = 0;

VkBuffer point_lights_buffer = VK_NULL_HANDLE;
VmaAllocation point_lights_buffer_allocation = VK_NULL_HANDLE;
void* point_lights_buffer_mapped = nullptr;
VkBuffer spot_lights_buffer = VK_NULL_HANDLE;
VmaAllocation spot_lights_buffer_allocation = VK_NULL_HANDLE;
void* spot_lights_buffer_mapped = nullptr;

std::vector<Texture> textures;
std::array<ModelInstance, model_count> models = {{
    {"Icosahedron 1", {-1.9f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
     {0.78f, 0.78f, 0.78f}, {1.0f, 1.0f, 1.0f}, 32.0f, 0, false, 2.0f, 0.5f},
    {"Icosahedron 2", {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
     {0.78f, 0.78f, 0.78f}, {1.0f, 1.0f, 1.0f}, 32.0f, 1, true, 2.0f, 0.75f},
    {"Icosahedron 3", {1.9f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
     {0.78f, 0.78f, 0.78f}, {1.0f, 1.0f, 1.0f}, 32.0f, 2, false, 2.0f, 0.5f},
}};

glm::vec3 camera_position(0.0f, 0.0f, 5.0f);
glm::vec3 camera_rotation(0.0f);
float camera_fov = 45.0f;
bool use_perspective = true;

float ambient_strength = 0.12f;
bool directional_enabled = true;
glm::vec3 directional_direction(-0.5f, -1.0f, -0.5f);
glm::vec3 directional_color(1.0f);
float directional_intensity = 1.0f;

bool animation_enabled = false;
float animation_speed = 1.0f;
float animation_radius = 1.0f;
float animation_time = 0.0f;
double last_time = 0.0;

std::vector<PointLight> point_lights = {
    {glm::vec4(2.0f, 1.0f, 2.0f, 18.0f), glm::vec4(1.0f, 0.25f, 0.15f, 1.0f)}
};
std::vector<SpotLight> spot_lights = {
    {glm::vec4(-2.0f, 2.0f, 2.0f, 24.0f),
     glm::vec4(0.0f, -0.6f, -1.0f, std::cos(glm::radians(15.0f))),
     glm::vec4(0.2f, 0.45f, 1.0f, std::cos(glm::radians(23.0f)))}
};

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return {};
    }

    const std::streamsize fileSize = file.tellg();
    if (fileSize <= 0) {
        std::cerr << "File is empty or cannot be read: " << filename << '\n';
        return {};
    }
    std::vector<char> buffer(static_cast<size_t>(fileSize));
    file.seekg(0);
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Failed to read file: " << filename << '\n';
        return {};
    }
    return buffer;
}

bool createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VmaMemoryUsage memoryUsage,
    VkBuffer& buffer,
    VmaAllocation& allocation,
    void** mappedData = nullptr
) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = memoryUsage;
    if (mappedData != nullptr) {
        allocationInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VmaAllocationInfo resultInfo{};
    if (vmaCreateBuffer(
            graphics::internal::context.allocator,
            &bufferInfo,
            &allocationInfo,
            &buffer,
            &allocation,
            &resultInfo) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return false;
    }

    if (mappedData != nullptr) {
        *mappedData = resultInfo.pMappedData;
    }
    return true;
}

template<typename RecordCommands>
bool submitImmediate(RecordCommands&& recordCommands) {
    auto& context = graphics::internal::context;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = context.graphics_queue_index;

    if (vkCreateCommandPool(context.device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        std::cerr << "Failed to create a temporary Vulkan command pool\n";
        return false;
    }

    VkCommandBufferAllocateInfo allocationInfo{};
    allocationInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocationInfo.commandPool = commandPool;
    allocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocationInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(context.device, &allocationInfo, &commandBuffer) != VK_SUCCESS) {
        std::cerr << "Failed to allocate a temporary Vulkan command buffer\n";
        vkDestroyCommandPool(context.device, commandPool, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        std::cerr << "Failed to begin a temporary Vulkan command buffer\n";
        vkDestroyCommandPool(context.device, commandPool, nullptr);
        return false;
    }

    recordCommands(commandBuffer);
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        std::cerr << "Failed to end a temporary Vulkan command buffer\n";
        vkDestroyCommandPool(context.device, commandPool, nullptr);
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    const bool submitted =
        vkQueueSubmit(context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS;
    const bool completed = submitted && vkQueueWaitIdle(context.graphics_queue) == VK_SUCCESS;
    vkDestroyCommandPool(context.device, commandPool, nullptr);
    if (!completed) {
        std::cerr << "Failed to submit or complete a Vulkan resource upload\n";
        return false;
    }
    return true;
}

bool createGpuBuffer(
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer& buffer,
    VmaAllocation& allocation
) {
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    if (!createBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU,
            stagingBuffer,
            stagingAllocation)) {
        return false;
    }

    void* mappedData = nullptr;
    if (vmaMapMemory(
            graphics::internal::context.allocator,
            stagingAllocation,
            &mappedData) != VK_SUCCESS) {
        std::cerr << "Failed to map a Vulkan staging buffer\n";
        vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
        return false;
    }
    std::memcpy(mappedData, data, static_cast<size_t>(size));
    if (vmaFlushAllocation(
            graphics::internal::context.allocator,
            stagingAllocation,
            0,
            size) != VK_SUCCESS) {
        std::cerr << "Failed to flush a Vulkan staging buffer\n";
        vmaUnmapMemory(graphics::internal::context.allocator, stagingAllocation);
        vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
        return false;
    }
    vmaUnmapMemory(graphics::internal::context.allocator, stagingAllocation);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateBuffer(
            graphics::internal::context.allocator,
            &bufferInfo,
            &allocationInfo,
            &buffer,
            &allocation,
            nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create a GPU-local model buffer\n";
        vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
        return false;
    }

    const bool copied = submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer, 1, &copy);
    });

    vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
    if (!copied) {
        vmaDestroyBuffer(graphics::internal::context.allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool loadTexture(const std::filesystem::path& path, Texture& texture) {
    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &sourceChannels, STBI_rgb_alpha);
    if (pixels == nullptr) {
        std::cerr << "Failed to load texture '" << path.string() << "': "
                  << stbi_failure_reason() << '\n';
        return false;
    }

    if (width <= 0 || height <= 0) {
        std::cerr << "Texture has invalid dimensions: " << path.string() << '\n';
        stbi_image_free(pixels);
        return false;
    }

    const VkDeviceSize imageSize =
        static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    if (!createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU,
            stagingBuffer,
            stagingAllocation)) {
        stbi_image_free(pixels);
        return false;
    }

    void* mappedData = nullptr;
    if (vmaMapMemory(
            graphics::internal::context.allocator,
            stagingAllocation,
            &mappedData) != VK_SUCCESS) {
        std::cerr << "Failed to map the staging buffer for texture: " << path.string() << '\n';
        stbi_image_free(pixels);
        vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
        return false;
    }
    std::memcpy(mappedData, pixels, static_cast<size_t>(imageSize));
    if (vmaFlushAllocation(
            graphics::internal::context.allocator,
            stagingAllocation,
            0,
            imageSize) != VK_SUCCESS) {
        std::cerr << "Failed to flush texture staging buffer: " << path.string() << '\n';
        vmaUnmapMemory(graphics::internal::context.allocator, stagingAllocation);
        stbi_image_free(pixels);
        vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
        return false;
    }
    vmaUnmapMemory(graphics::internal::context.allocator, stagingAllocation);
    stbi_image_free(pixels);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1
    };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo imageAllocationInfo{};
    imageAllocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(
            graphics::internal::context.allocator,
            &imageInfo,
            &imageAllocationInfo,
            &texture.image,
            &texture.allocation,
            nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan image for texture: " << path.string() << '\n';
        vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
        return false;
    }

    const bool uploaded = submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier toTransfer{};
        toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = texture.image;
        toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toTransfer.subresourceRange.levelCount = 1;
        toTransfer.subresourceRange.layerCount = 1;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toTransfer);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = imageInfo.extent;
        vkCmdCopyBufferToImage(
            commandBuffer,
            stagingBuffer,
            texture.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy);

        VkImageMemoryBarrier toShader{};
        toShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toShader.image = texture.image;
        toShader.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toShader.subresourceRange.levelCount = 1;
        toShader.subresourceRange.layerCount = 1;
        toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toShader);
    });
    vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
    if (!uploaded) {
        vmaDestroyImage(
            graphics::internal::context.allocator,
            texture.image,
            texture.allocation);
        texture.image = VK_NULL_HANDLE;
        texture.allocation = VK_NULL_HANDLE;
        return false;
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = texture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(
            graphics::internal::context.device,
            &viewInfo,
            nullptr,
            &texture.view) != VK_SUCCESS) {
        std::cerr << "Failed to create image view for texture: " << path.string() << '\n';
        return false;
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.maxLod = 0.0f;
    if (vkCreateSampler(
            graphics::internal::context.device,
            &samplerInfo,
            nullptr,
            &texture.sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create texture sampler: " << path.string() << '\n';
        return false;
    }

    texture.name = path.stem().string();
    return true;
}

bool loadTextures() {
    const std::filesystem::path textureDirectory("textures");
    std::error_code error;
    if (!std::filesystem::exists(textureDirectory, error) || error ||
        !std::filesystem::is_directory(textureDirectory, error) || error) {
        std::cerr << "Texture directory 'textures' is missing or cannot be accessed\n";
        return false;
    }

    const std::array<std::filesystem::path, 3> paths = {
        textureDirectory / "image1.png",
        textureDirectory / "image2.png",
        textureDirectory / "image3.png"
    };

    textures.reserve(paths.size());
    for (const auto& path : paths) {
        textures.emplace_back();
        if (!loadTexture(path, textures.back())) {
            return false;
        }
    }

    for (auto& model : models) {
        model.textureIndex %= textures.size();
    }
    return true;
}

void generateIcosahedron(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    const std::array<glm::vec3, 12> positions = {{
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}
    }};
    const std::array<uint32_t, 60> faces = {{
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
        1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
        3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
        4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
    }};

    vertices.clear();
    indices.clear();
    vertices.reserve(faces.size());
    indices.reserve(faces.size());

    for (size_t face = 0; face < faces.size(); face += 3) {
        std::array<glm::vec2, 3> uv{};
        for (size_t corner = 0; corner < 3; ++corner) {
            const glm::vec3 normal = glm::normalize(positions[faces[face + corner]]);
            uv[corner] = {
                0.5f + std::atan2(normal.z, normal.x) / (2.0f * glm::pi<float>()),
                0.5f - std::asin(normal.y) / glm::pi<float>()
            };
        }
        const float minimumU = std::min({uv[0].x, uv[1].x, uv[2].x});
        const float maximumU = std::max({uv[0].x, uv[1].x, uv[2].x});
        if (maximumU - minimumU > 0.5f) {
            for (auto& texCoord : uv) {
                if (texCoord.x < 0.5f) {
                    texCoord.x += 1.0f;
                }
            }
        }

        for (size_t corner = 0; corner < 3; ++corner) {
            const glm::vec3 position = positions[faces[face + corner]];
            vertices.push_back({position, glm::normalize(position), uv[corner]});
            indices.push_back(static_cast<uint32_t>(indices.size()));
        }
    }
}

glm::mat4 makeModelMatrix(const ModelInstance& instance, size_t index) {
    glm::vec3 position = instance.position;
    glm::vec3 rotation = instance.rotation;
    if (animation_enabled && index == 0) {
        position.x += animation_radius * std::cos(animation_time);
        position.y += animation_radius * std::sin(animation_time);
        rotation.z += glm::degrees(animation_time * 0.8f);
    }

    glm::mat4 model(1.0f);
    model = glm::translate(model, position);
    model = glm::rotate(model, glm::radians(rotation.x), {1.0f, 0.0f, 0.0f});
    model = glm::rotate(model, glm::radians(rotation.y), {0.0f, 1.0f, 0.0f});
    model = glm::rotate(model, glm::radians(rotation.z), {0.0f, 0.0f, 1.0f});
    return glm::scale(model, instance.scale);
}

void drawLightingControls() {
    if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable directional light", &directional_enabled);
        ImGui::SliderFloat3("Direction", &directional_direction.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", &directional_color.x);
        ImGui::SliderFloat("Intensity", &directional_intensity, 0.0f, 5.0f);
        ImGui::SliderFloat("Ambient light", &ambient_strength, 0.0f, 1.0f);
    }

    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (point_lights.size() < max_lights && ImGui::Button("Add point light")) {
            point_lights.push_back({
                glm::vec4(1.0f, 1.0f, 2.0f, 15.0f),
                glm::vec4(1.0f, 0.8f, 0.5f, 1.0f)
            });
        }
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            const std::string label = "Point " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                PointLight& light = point_lights[i];
                ImGui::SliderFloat3("Position", &light.positionIntensity.x, -5.0f, 5.0f);
                ImGui::SliderFloat("Intensity", &light.positionIntensity.w, 0.0f, 100.0f);
                ImGui::ColorEdit3("Color", &light.color.x);
                if (ImGui::Button("Remove")) {
                    point_lights.erase(point_lights.begin() + static_cast<std::ptrdiff_t>(i));
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
        if (spot_lights.size() < max_lights && ImGui::Button("Add spot light")) {
            spot_lights.push_back({
                glm::vec4(0.0f, 2.0f, 2.0f, 25.0f),
                glm::vec4(0.0f, -1.0f, -1.0f, std::cos(glm::radians(15.0f))),
                glm::vec4(1.0f, 1.0f, 0.8f, std::cos(glm::radians(25.0f)))
            });
        }
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(1000 + static_cast<int>(i));
            const std::string label = "Spot " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                SpotLight& light = spot_lights[i];
                ImGui::SliderFloat3("Position", &light.positionIntensity.x, -5.0f, 5.0f);
                ImGui::SliderFloat("Intensity", &light.positionIntensity.w, 0.0f, 100.0f);
                ImGui::SliderFloat3("Direction", &light.directionInner.x, -1.0f, 1.0f);
                float inner = glm::degrees(std::acos(glm::clamp(light.directionInner.w, -1.0f, 1.0f)));
                float outer = glm::degrees(std::acos(glm::clamp(light.colorOuter.w, -1.0f, 1.0f)));
                if (ImGui::SliderFloat("Inner angle", &inner, 1.0f, 60.0f)) {
                    light.directionInner.w = std::cos(glm::radians(inner));
                }
                if (ImGui::SliderFloat("Outer angle", &outer, inner, 75.0f)) {
                    light.colorOuter.w = std::cos(glm::radians(std::max(outer, inner)));
                }
                if (outer < inner) {
                    light.colorOuter.w = std::cos(glm::radians(inner));
                }
                ImGui::ColorEdit3("Color", &light.colorOuter.x);
                if (ImGui::Button("Remove")) {
                    spot_lights.erase(spot_lights.begin() + static_cast<std::ptrdiff_t>(i));
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }
}

} // namespace

namespace application {

bool initialize() {
    auto& context = graphics::internal::context;

    if (!loadTextures()) {
        return false;
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    generateIcosahedron(vertices, indices);
    index_count = static_cast<uint32_t>(indices.size());
    if (!createGpuBuffer(
            vertices.data(),
            sizeof(Vertex) * vertices.size(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertex_buffer,
            vertex_buffer_allocation) ||
        !createGpuBuffer(
            indices.data(),
            sizeof(uint32_t) * indices.size(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            index_buffer,
            index_buffer_allocation)) {
        return false;
    }

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(context.physical_device, &deviceProperties);
    const VkDeviceSize uniformAlignment =
        std::max<VkDeviceSize>(1, deviceProperties.limits.minUniformBufferOffsetAlignment);
    if (sizeof(ModelUniformBufferObject) > deviceProperties.limits.maxUniformBufferRange) {
        std::cerr << "Model uniform data exceeds the device's maximum uniform buffer range\n";
        return false;
    }
    model_uniform_stride =
        (sizeof(ModelUniformBufferObject) + uniformAlignment - 1) / uniformAlignment *
        uniformAlignment;
    if (model_uniform_stride * (model_count - 1) >
        std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Dynamic uniform buffer offsets exceed Vulkan's 32-bit limit\n";
        return false;
    }

    if (!createBuffer(
            sizeof(SceneUniformBufferObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU,
            scene_uniform_buffer,
            scene_uniform_buffer_allocation,
            &scene_uniform_buffer_mapped) ||
        !createBuffer(
            model_uniform_stride * model_count,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU,
            model_uniform_buffer,
            model_uniform_buffer_allocation,
            &model_uniform_buffer_mapped) ||
        !createBuffer(
            sizeof(PointLight) * max_lights,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU,
            point_lights_buffer,
            point_lights_buffer_allocation,
            &point_lights_buffer_mapped) ||
        !createBuffer(
            sizeof(SpotLight) * max_lights,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU,
            spot_lights_buffer,
            spot_lights_buffer_allocation,
            &spot_lights_buffer_mapped)) {
        return false;
    }

    const std::array<VkDescriptorSetLayoutBinding, 3> sceneBindings = {{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    }};
    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
    sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutInfo.bindingCount = static_cast<uint32_t>(sceneBindings.size());
    sceneLayoutInfo.pBindings = sceneBindings.data();
    if (vkCreateDescriptorSetLayout(
            context.device, &sceneLayoutInfo, nullptr, &scene_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create scene descriptor set layout\n";
        return false;
    }

    const std::array<VkDescriptorSetLayoutBinding, 2> materialBindings = {{
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
    }};
    VkDescriptorSetLayoutCreateInfo materialLayoutInfo{};
    materialLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    materialLayoutInfo.bindingCount = static_cast<uint32_t>(materialBindings.size());
    materialLayoutInfo.pBindings = materialBindings.data();
    if (vkCreateDescriptorSetLayout(
            context.device,
            &materialLayoutInfo,
            nullptr,
            &material_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create material descriptor set layout\n";
        return false;
    }

    const std::array<VkDescriptorPoolSize, 3> poolSizes = {{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, static_cast<uint32_t>(textures.size())},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(textures.size())}
    }};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = static_cast<uint32_t>(textures.size() + 1);
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    VkDescriptorSetAllocateInfo sceneSetAllocation{};
    sceneSetAllocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sceneSetAllocation.descriptorPool = descriptor_pool;
    sceneSetAllocation.descriptorSetCount = 1;
    sceneSetAllocation.pSetLayouts = &scene_descriptor_set_layout;
    if (vkAllocateDescriptorSets(context.device, &sceneSetAllocation, &scene_descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate scene descriptor set\n";
        return false;
    }

    std::vector<VkDescriptorSetLayout> materialLayouts(textures.size(), material_descriptor_set_layout);
    material_descriptor_sets.resize(textures.size());
    VkDescriptorSetAllocateInfo materialSetAllocation{};
    materialSetAllocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    materialSetAllocation.descriptorPool = descriptor_pool;
    materialSetAllocation.descriptorSetCount = static_cast<uint32_t>(materialLayouts.size());
    materialSetAllocation.pSetLayouts = materialLayouts.data();
    if (vkAllocateDescriptorSets(
            context.device,
            &materialSetAllocation,
            material_descriptor_sets.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate material descriptor sets\n";
        return false;
    }

    const std::array<VkDescriptorBufferInfo, 3> sceneBufferInfos = {{
        {scene_uniform_buffer, 0, sizeof(SceneUniformBufferObject)},
        {point_lights_buffer, 0, sizeof(PointLight) * max_lights},
        {spot_lights_buffer, 0, sizeof(SpotLight) * max_lights}
    }};
    std::array<VkWriteDescriptorSet, 3> sceneWrites{};
    for (uint32_t binding = 0; binding < sceneWrites.size(); ++binding) {
        sceneWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sceneWrites[binding].dstSet = scene_descriptor_set;
        sceneWrites[binding].dstBinding = binding;
        sceneWrites[binding].descriptorCount = 1;
        sceneWrites[binding].descriptorType = binding == 0
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sceneWrites[binding].pBufferInfo = &sceneBufferInfos[binding];
    }
    vkUpdateDescriptorSets(
        context.device,
        static_cast<uint32_t>(sceneWrites.size()),
        sceneWrites.data(),
        0,
        nullptr);

    for (size_t i = 0; i < textures.size(); ++i) {
        const VkDescriptorBufferInfo modelBufferInfo{
            model_uniform_buffer, 0, sizeof(ModelUniformBufferObject)
        };
        const VkDescriptorImageInfo imageInfo{
            textures[i].sampler,
            textures[i].view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        };
        const std::array<VkWriteDescriptorSet, 2> writes = {{
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material_descriptor_sets[i],
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &modelBufferInfo
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material_descriptor_sets[i],
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfo
            }
        }};
        vkUpdateDescriptorSets(
            context.device,
            static_cast<uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);
    }

    const VkDescriptorSetLayout pipelineLayouts[] = {
        scene_descriptor_set_layout,
        material_descriptor_set_layout
    };
    const VkPushConstantRange pushConstantRange{
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(SamplingSettings)
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = pipelineLayouts;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(
            context.device, &pipelineLayoutInfo, nullptr, &pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline layout\n";
        return false;
    }

    const std::vector<char> vertCode = readFile("shaders/icosahedron.vert.spv");
    const std::vector<char> fragCode = readFile("shaders/icosahedron.frag.spv");
    if (vertCode.empty() || fragCode.empty()) {
        return false;
    }
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;

    VkShaderModuleCreateInfo vertInfo{};
    vertInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vertInfo.codeSize = vertCode.size();
    vertInfo.pCode = reinterpret_cast<const uint32_t*>(vertCode.data());
    if (vkCreateShaderModule(context.device, &vertInfo, nullptr, &vertModule) != VK_SUCCESS) {
        std::cerr << "Failed to create vertex shader module\n";
        return false;
    }

    VkShaderModuleCreateInfo fragInfo{};
    fragInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fragInfo.codeSize = fragCode.size();
    fragInfo.pCode = reinterpret_cast<const uint32_t*>(fragCode.data());
    if (vkCreateShaderModule(context.device, &fragInfo, nullptr, &fragModule) != VK_SUCCESS) {
        std::cerr << "Failed to create fragment shader module\n";
        vkDestroyShaderModule(context.device, vertModule, nullptr);
        return false;
    }

    const VkPipelineShaderStageCreateInfo shaderStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertModule,
            .pName = "main"
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragModule,
            .pName = "main"
        }
    };

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    const std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)}
    }};
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipeline_layout;
    pipelineInfo.renderPass = context.render_pass;
    if (vkCreateGraphicsPipelines(
            context.device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &graphics_pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        vkDestroyShaderModule(context.device, fragModule, nullptr);
        vkDestroyShaderModule(context.device, vertModule, nullptr);
        return false;
    }

    vkDestroyShaderModule(context.device, fragModule, nullptr);
    vkDestroyShaderModule(context.device, vertModule, nullptr);
    last_time = glfwGetTime();
    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    if (context.device == VK_NULL_HANDLE) {
        return;
    }
    vkQueueWaitIdle(context.graphics_queue);

    if (graphics_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
        graphics_pipeline = VK_NULL_HANDLE;
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
        pipeline_layout = VK_NULL_HANDLE;
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
        descriptor_pool = VK_NULL_HANDLE;
    }
    if (scene_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device, scene_descriptor_set_layout, nullptr);
        scene_descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (material_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device, material_descriptor_set_layout, nullptr);
        material_descriptor_set_layout = VK_NULL_HANDLE;
    }

    for (auto& texture : textures) {
        if (texture.sampler != VK_NULL_HANDLE) {
            vkDestroySampler(context.device, texture.sampler, nullptr);
        }
        if (texture.view != VK_NULL_HANDLE) {
            vkDestroyImageView(context.device, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE) {
            vmaDestroyImage(context.allocator, texture.image, texture.allocation);
        }
    }
    textures.clear();
    material_descriptor_sets.clear();

    const auto destroyBuffer = [&](VkBuffer& buffer, VmaAllocation& allocation) {
        if (buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, buffer, allocation);
            buffer = VK_NULL_HANDLE;
            allocation = VK_NULL_HANDLE;
        }
    };
    destroyBuffer(scene_uniform_buffer, scene_uniform_buffer_allocation);
    destroyBuffer(model_uniform_buffer, model_uniform_buffer_allocation);
    destroyBuffer(point_lights_buffer, point_lights_buffer_allocation);
    destroyBuffer(spot_lights_buffer, spot_lights_buffer_allocation);
    destroyBuffer(index_buffer, index_buffer_allocation);
    destroyBuffer(vertex_buffer, vertex_buffer_allocation);
}

void update(double time) {
    const double delta = std::clamp(time - last_time, 0.0, 0.1);
    last_time = time;
    if (animation_enabled) {
        animation_time += static_cast<float>(delta) * animation_speed;
    }

    ImGui::Begin("Lab 3 - Texture mapping");
    if (ImGui::CollapsingHeader("Models and materials", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < models.size(); ++i) {
            ModelInstance& model = models[i];
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::TreeNode(model.name)) {
                if (ImGui::BeginCombo(
                        "Albedo texture",
                        textures[model.textureIndex].name.c_str())) {
                    for (size_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
                        const bool selected = model.textureIndex == textureIndex;
                        if (ImGui::Selectable(textures[textureIndex].name.c_str(), selected)) {
                            model.textureIndex = textureIndex;
                        }
                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::SliderFloat3("Position", &model.position.x, -4.0f, 4.0f);
                ImGui::SliderFloat3("Rotation", &model.rotation.x, -180.0f, 180.0f);
                ImGui::SliderFloat3("Scale", &model.scale.x, 0.2f, 1.5f);
                ImGui::ColorEdit3("Albedo tint", &model.albedo.x);
                ImGui::SliderFloat("Shininess", &model.shininess, 1.0f, 128.0f);
                if (i == 1) {
                    ImGui::Separator();
                    ImGui::Checkbox("Animated texture sampling", &model.animatedSampling);
                    ImGui::SliderFloat("UV scale", &model.uvScale, 0.5f, 8.0f);
                    ImGui::SliderFloat("Sample blend", &model.blendFactor, 0.0f, 1.0f);
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::Separator();
        ImGui::Checkbox("Animate first model", &animation_enabled);
        if (animation_enabled) {
            ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 5.0f);
            ImGui::SliderFloat("Animation radius", &animation_radius, 0.0f, 2.0f);
        }
    }

    if (ImGui::CollapsingHeader("Camera")) {
        ImGui::Checkbox("Perspective projection", &use_perspective);
        ImGui::SliderFloat3("Position", &camera_position.x, -10.0f, 10.0f);
        ImGui::SliderFloat2("Rotation (pitch, yaw)", &camera_rotation.x, -89.0f, 89.0f);
        if (use_perspective) {
            ImGui::SliderFloat("Field of view", &camera_fov, 20.0f, 100.0f);
        }
    }

    drawLightingControls();
    ImGui::End();

    glm::vec3 forward;
    forward.x = std::cos(glm::radians(camera_rotation.x)) *
        std::sin(glm::radians(camera_rotation.y));
    forward.y = std::sin(glm::radians(camera_rotation.x));
    forward.z = -std::cos(glm::radians(camera_rotation.x)) *
        std::cos(glm::radians(camera_rotation.y));
    const glm::mat4 view = glm::lookAt(
        camera_position,
        camera_position + forward,
        glm::vec3(0.0f, 1.0f, 0.0f));

    const auto& extent = graphics::internal::context.swapchain_extent;
    const float aspect = static_cast<float>(extent.width) /
        static_cast<float>(std::max(1u, extent.height));
    glm::mat4 projection;
    if (use_perspective) {
        projection = glm::perspectiveRH_ZO(glm::radians(camera_fov), aspect, 0.1f, 100.0f);
    } else {
        projection = glm::orthoRH_ZO(
            -3.0f * aspect, 3.0f * aspect, -3.0f, 3.0f, 0.1f, 100.0f);
    }
    projection[1][1] *= -1.0f;

    scene_ubo.view = view;
    scene_ubo.proj = projection;
    scene_ubo.cameraPos = glm::vec4(camera_position, 1.0f);
    scene_ubo.dirDirection = glm::vec4(glm::normalize(directional_direction), 0.0f);
    scene_ubo.dirColorIntensity = glm::vec4(directional_color, directional_intensity);
    scene_ubo.ambient = glm::vec4(glm::vec3(ambient_strength), 1.0f);
    scene_ubo.counts = glm::ivec4(
        static_cast<int>(point_lights.size()),
        static_cast<int>(spot_lights.size()),
        directional_enabled ? 1 : 0,
        0);

    for (size_t i = 0; i < models.size(); ++i) {
        const ModelInstance& instance = models[i];
        model_ubos[i].model = makeModelMatrix(instance, i);
        model_ubos[i].albedo = glm::vec4(instance.albedo, 1.0f);
        model_ubos[i].specular = glm::vec4(1.0f);
        model_ubos[i].material = glm::vec4(instance.shininess, 0.0f, 0.0f, 0.0f);
        std::memcpy(
            static_cast<std::byte*>(model_uniform_buffer_mapped) + model_uniform_stride * i,
            &model_ubos[i],
            sizeof(ModelUniformBufferObject));
    }

    std::memcpy(scene_uniform_buffer_mapped, &scene_ubo, sizeof(scene_ubo));
    if (!point_lights.empty()) {
        std::memcpy(
            point_lights_buffer_mapped,
            point_lights.data(),
            sizeof(PointLight) * point_lights.size());
    }
    if (!spot_lights.empty()) {
        std::memcpy(
            spot_lights_buffer_mapped,
            spot_lights.data(),
            sizeof(SpotLight) * spot_lights.size());
    }

    auto& context = graphics::internal::context;
    if (vmaFlushAllocation(
            context.allocator, scene_uniform_buffer_allocation, 0, sizeof(scene_ubo)) != VK_SUCCESS ||
        vmaFlushAllocation(
            context.allocator,
            model_uniform_buffer_allocation,
            0,
            model_uniform_stride * model_count) != VK_SUCCESS ||
        vmaFlushAllocation(
            context.allocator,
            point_lights_buffer_allocation,
            0,
            sizeof(PointLight) * max_lights) != VK_SUCCESS ||
        vmaFlushAllocation(
            context.allocator,
            spot_lights_buffer_allocation,
            0,
            sizeof(SpotLight) * max_lights) != VK_SUCCESS) {
        std::cerr << "Failed to flush one or more per-frame Vulkan buffers\n";
    }
}

void render(const graphics::internal::FrameData& frame) {
    auto& context = graphics::internal::context;
    vkResetCommandBuffer(frame.command_buffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(frame.command_buffer, &beginInfo);

    VkClearValue clearValues[2]{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = context.render_pass;
    renderPassInfo.framebuffer = frame.framebuffer;
    renderPassInfo.renderArea.extent = context.swapchain_extent;
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(frame.command_buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

    VkViewport viewport{};
    viewport.width = static_cast<float>(context.swapchain_extent.width);
    viewport.height = static_cast<float>(context.swapchain_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(frame.command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = context.swapchain_extent;
    vkCmdSetScissor(frame.command_buffer, 0, 1, &scissor);

    VkBuffer vertexBuffers[] = {vertex_buffer};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(frame.command_buffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(frame.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(
        frame.command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout,
        0,
        1,
        &scene_descriptor_set,
        0,
        nullptr);

    for (size_t i = 0; i < models.size(); ++i) {
        const ModelInstance& model = models[i];
        const uint32_t dynamicOffset = static_cast<uint32_t>(model_uniform_stride * i);
        const VkDescriptorSet materialSet = material_descriptor_sets[model.textureIndex];
        vkCmdBindDescriptorSets(
            frame.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            1,
            1,
            &materialSet,
            1,
            &dynamicOffset);

        const SamplingSettings sampling{
            static_cast<float>(glfwGetTime()),
            model.uvScale,
            model.animatedSampling ? 1 : 0,
            model.blendFactor
        };
        vkCmdPushConstants(
            frame.command_buffer,
            pipeline_layout,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(sampling),
            &sampling);
        vkCmdDrawIndexed(frame.command_buffer, index_count, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(frame.command_buffer);
    vkEndCommandBuffer(frame.command_buffer);
}

} // namespace application
