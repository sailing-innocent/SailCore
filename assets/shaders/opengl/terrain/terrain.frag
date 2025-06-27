#version 450 core
out vec4 FragColor;

uniform vec4 lightColor;
uniform vec4 lightPos;
uniform vec3 viewPos;

in vec3 FragPos;
in vec3 Normal;

void main() {
    FragColor = vec4(1.0, 1.0f, 1.0f, 1.0f);
}
