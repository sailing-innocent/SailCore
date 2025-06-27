#version 450 core 
out vec4 FragColor;

uniform vec4 lightColor;
uniform vec4 lightPos;
uniform vec3 viewPos;

in vec3 FragPos;
in vec3 Normal;

void main() {
    vec3 objectColor = vec3(1.0, 0.5, 0.31);
    vec3 normal = normalize(Normal);

    vec3 lightDir = normalize(vec3(lightPos) - FragPos);
    // ambient 
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * vec3(lightColor);

    // diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vec3(lightColor);
    
    // specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(vec3(viewPos) - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);

    vec3 specular = specularStrength * spec * vec3(lightColor);

    // vec3 result = objectColor; // original
    // vec3 result = ambient * objectColor; // ambient only
    // vec3 result = (ambient + diffuse) * objectColor; // ambient + diffuse
    vec3 result = (ambient + diffuse + specular) * objectColor; // ambient + diffuse + specular

    FragColor = vec4(result, 1.0);
}