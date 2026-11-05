#version 450
layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 texCoord;
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

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight lights[];
} points;

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight lights[];
} spots;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 material;
} objectData;

layout(set = 1, binding = 1) uniform sampler2D diffuseTexture;

layout(push_constant) uniform SamplingSettings {
    float time;
    float uvScale;
    int enabled;
    float blendFactor;
} sampling;

vec3 blinnPhong(
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 lightColor,
    float intensity,
    vec3 baseColor
) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);

    float spec = diff > 0.0
        ? pow(
            max(dot(N, H), 0.0),
            max(objectData.material.x, 1.0)
        )
        : 0.0;

    return intensity * lightColor *
        (baseColor * diff + objectData.specular.rgb * spec);
}

vec3 sampleBaseTexture(vec2 uv) {
    if (sampling.enabled == 0) {
        return texture(diffuseTexture, uv).rgb;
    }

    float scale = max(sampling.uvScale, 0.1);
    vec2 animatedUV = uv * scale + vec2(
        sin(sampling.time * 0.7 + uv.y * 6.28318),
        cos(sampling.time * 0.9 + uv.x * 6.28318)
    ) * 0.12;
    vec2 secondaryUV = uv * scale + vec2(
        cos(sampling.time * 0.5 + uv.x * 6.28318),
        sin(sampling.time * 0.8 + uv.y * 6.28318)
    ) * 0.18;

    vec3 firstSample = texture(diffuseTexture, animatedUV).rgb;
    vec3 secondSample = texture(diffuseTexture, secondaryUV).rgb;
    vec3 modulated = mix(firstSample, secondSample, 0.5);
    vec3 tint = 0.7 + 0.3 * cos(
        sampling.time + vec3(0.0, 2.0944, 4.18879)
    );
    return mix(firstSample, modulated * tint, clamp(sampling.blendFactor, 0.0, 1.0));
}

void main() {
    vec3 textureColor = sampleBaseTexture(texCoord);
    vec3 baseColor = textureColor * objectData.albedo.rgb;

    vec3 N = normalize(worldNormal);
    vec3 V = normalize(scene.cameraPos.xyz - worldPosition);

    vec3 result = scene.ambient.rgb * baseColor;

    if (scene.counts.z != 0) {
        vec3 L = normalize(-scene.dirDirection.xyz);

        result += blinnPhong(
            N, V, L,
            scene.dirColorIntensity.rgb,
            scene.dirColorIntensity.w,
            baseColor
        );
    }

    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 delta =
            points.lights[i].positionIntensity.xyz - worldPosition;

        float d2 = max(dot(delta, delta), 0.01);
        vec3 L = delta * inversesqrt(d2);

        result += blinnPhong(
            N, V, L,
            points.lights[i].color.rgb,
            points.lights[i].positionIntensity.w / d2,
            baseColor
        );
    }

    for (int i = 0; i < scene.counts.y; ++i) {
        SpotLight light = spots.lights[i];

        vec3 delta = light.positionIntensity.xyz - worldPosition;
        float d2 = max(dot(delta, delta), 0.01);
        vec3 L = delta * inversesqrt(d2);

        vec3 fromLightToPoint = -L;

        float theta = dot(
            normalize(light.directionInner.xyz),
            fromLightToPoint
        );

        float cone = smoothstep(
            light.colorOuter.w,
            light.directionInner.w,
            theta
        );

        result += cone * blinnPhong(
            N, V, L,
            light.colorOuter.rgb,
            light.positionIntensity.w / d2,
            baseColor
        );
    }

    outColor = vec4(result, 1.0);
}
