#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 dirDirection;
    vec4 dirColorIntensity;
    vec4 ambient;
    ivec4 counts;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 material;
} objectData;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

void main() {
    vec4 world = objectData.model * vec4(inPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = normalize(transpose(inverse(mat3(objectData.model))) * inNormal);
    gl_Position = scene.proj * scene.view * world;
}
