#version 450
layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 dirDirection;
    vec4 dirColorIntensity;
    vec4 ambient;
    ivec4 counts;
} scene;

struct PointLight {
    vec4 positionIntensity;
    vec4 color;
};
struct SpotLight {
    vec4 positionIntensity;
    vec4 directionInner;
    vec4 colorOuter;
};
layout(std430, set = 0, binding = 1) readonly buffer PointLights { PointLight lights[]; } points;
layout(std430, set = 0, binding = 2) readonly buffer SpotLights { SpotLight lights[]; } spots;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 material;
} objectData;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 lightColor, float intensity) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = diff > 0.0 ? pow(max(dot(N, H), 0.0), max(objectData.material.x, 1.0)) : 0.0;
    return intensity * lightColor * (objectData.albedo.rgb * diff + objectData.specular.rgb * spec);
}

void main() {
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(scene.cameraPos.xyz - worldPosition);
    vec3 result = scene.ambient.rgb * objectData.albedo.rgb;

    if (scene.counts.z != 0) {
        vec3 L = normalize(-scene.dirDirection.xyz);
        result += blinnPhong(N, V, L, scene.dirColorIntensity.rgb, scene.dirColorIntensity.w);
    }

    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 delta = points.lights[i].positionIntensity.xyz - worldPosition;
        float d2 = max(dot(delta, delta), 0.01);
        vec3 L = delta * inversesqrt(d2);
        result += blinnPhong(N, V, L, points.lights[i].color.rgb,
                             points.lights[i].positionIntensity.w / d2);
    }

    for (int i = 0; i < scene.counts.y; ++i) {
        SpotLight light = spots.lights[i];
        vec3 delta = light.positionIntensity.xyz - worldPosition;
        float d2 = max(dot(delta, delta), 0.01);
        vec3 L = delta * inversesqrt(d2);
        vec3 fromLightToPoint = -L;
        float theta = dot(normalize(light.directionInner.xyz), fromLightToPoint);
        float cone = smoothstep(light.colorOuter.w, light.directionInner.w, theta);
        result += cone * blinnPhong(N, V, L, light.colorOuter.rgb,
                                    light.positionIntensity.w / d2);
    }

    outColor = vec4(result, 1.0);
}
