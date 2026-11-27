#include "ui.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

#include <imgui.h>

namespace ui {

namespace {

void colorAndIntensity(glm::vec3& color, float& intensity) {
	ImGui::ColorEdit3("Color", &color.x);
	ImGui::DragFloat("Intensity", &intensity, 0.05f, 0.0f, 100.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
}

void direction(float& yaw, float& pitch) {
	ImGui::DragFloat("Yaw", &yaw, 0.5f, -180.0f, 180.0f, "%.0f deg", ImGuiSliderFlags_AlwaysClamp);
	ImGui::DragFloat("Pitch", &pitch, 0.5f, -90.0f, 90.0f, "%.0f deg", ImGuiSliderFlags_AlwaysClamp);
}

void cameraSection(scene::Camera& c) {
	if (!ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}
	ImGui::PushID("camera");
	if (ImGui::RadioButton("Perspective", c.projection == scene::Projection::Perspective)) {
		c.projection = scene::Projection::Perspective;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("Orthographic", c.projection == scene::Projection::Orthographic)) {
		c.projection = scene::Projection::Orthographic;
	}
	ImGui::DragFloat3("Position", &c.position.x, 0.02f);
	// при pitch = ±90 взгляд параллелен оси Y и lookAt вырождается
	ImGui::DragFloat("Yaw", &c.yaw_degrees, 0.5f, -180.0f, 180.0f, "%.0f deg", ImGuiSliderFlags_AlwaysClamp);
	ImGui::DragFloat("Pitch", &c.pitch_degrees, 0.5f, -89.0f, 89.0f, "%.0f deg", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Field of view", &c.fov_degrees, 20.0f, 120.0f, "%.0f deg");
	ImGui::PopID();
}

// лаба 1: выбор пирамиды, добавление и удаление
void objectsSection(scene::Scene& s) {
	if (!ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	char label[32];
	std::snprintf(label, sizeof(label), "Pyramid %d", s.selected + 1);
	if (ImGui::BeginCombo("Selected", label)) {
		for (int i = 0; i < s.count; ++i) {
			std::snprintf(label, sizeof(label), "Pyramid %d", i + 1);
			if (ImGui::Selectable(label, s.selected == i)) {
				s.selected = i;
			}
		}
		ImGui::EndCombo();
	}

	ImGui::BeginDisabled(s.count >= scene::max_objects);
	if (ImGui::Button("Add")) {
		scene::addObject(s);
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::BeginDisabled(s.count <= 1);
	if (ImGui::Button("Remove")) {
		scene::removeSelected(s);
	}
	ImGui::EndDisabled();
}

// лаба 1: трансформации
void transformSection(scene::Object& o) {
	if (!ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}
	// отрицательный масштаб выворачивает треугольники, грани пропадают при отсечении
	ImGui::DragFloat3("Position", &o.position.x, 0.01f, -5.0f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::DragFloat3("Rotation", &o.rotation_degrees.x, 0.5f, -180.0f, 180.0f, "%.0f deg",
		ImGuiSliderFlags_AlwaysClamp);
	ImGui::DragFloat3("Scale", &o.scale.x, 0.01f, 0.05f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
}

// лаба 1: анимация по траектории
void motionSection(scene::Motion& m) {
	if (!ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}
	if (ImGui::Button(m.playing ? "Pause###play" : "Play###play")) {
		m.playing = !m.playing;
	}
	ImGui::SliderFloat("Speed", &m.speed, 0.0f, 3.0f, "%.2f rad/s");
	ImGui::SliderFloat("Spin speed", &m.spin_speed_degrees, -360.0f, 360.0f, "%.0f deg/s");
	ImGui::SliderFloat("Radius", &m.radius, 0.0f, 3.0f, "%.2f");
	ImGui::SliderFloat("Wave height", &m.height, 0.0f, 2.0f, "%.2f");
	ImGui::SliderInt("Waves per lap", &m.waves, 1, 8);
	ImGui::TextDisabled("p(a) = (R cos a, H sin(k a), R sin a)");
}

// лаба 2: материал; лаба 3: альбедо — текстура, выбирается из выпадающего списка
void materialSection(scene::Object& o) {
	if (!ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}
	if (ImGui::BeginCombo("Texture", scene::materials[o.material.textures].name)) {
		for (int i = 0; i < scene::material_count; ++i) {
			if (ImGui::Selectable(scene::materials[i].name, o.material.textures == i)) {
				o.material.textures = i;
			}
		}
		ImGui::EndCombo();
	}
	ImGui::ColorEdit3("Specular color", &o.material.specular.x);
	ImGui::DragFloat("Shininess", &o.material.shininess, 0.5f, 1.0f, 256.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::Checkbox("Animated sampling", &o.animated_sampling);
}

void ambientAndDirectionalSection(scene::Scene& s) {
	if (!ImGui::CollapsingHeader("Ambient & directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}
	ImGui::PushID("ambient");
	ImGui::SeparatorText("Ambient");
	colorAndIntensity(s.ambient.color, s.ambient.intensity);
	ImGui::PopID();

	ImGui::PushID("directional");
	ImGui::SeparatorText("Directional");
	colorAndIntensity(s.directional.color, s.directional.intensity);
	direction(s.directional.yaw_degrees, s.directional.pitch_degrees);
	ImGui::PopID();
}

void pointLightsSection(std::vector<scene::PointLight>& lights) {
	if (!ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}
	ImGui::PushID("points");
	if (ImGui::Button("Add point light")) {
		lights.push_back({});
	}
	for (size_t i = 0; i < lights.size(); ++i) {
		ImGui::PushID(int(i));
		ImGui::SeparatorText(("Point light " + std::to_string(i + 1)).c_str());
		ImGui::DragFloat3("Position", &lights[i].position.x, 0.02f);
		colorAndIntensity(lights[i].color, lights[i].intensity);
		const bool remove = ImGui::Button("Remove");
		ImGui::PopID();
		if (remove) {
			lights.erase(lights.begin() + i);
			break; // список изменился, остальные покажем в следующем кадре
		}
	}
	ImGui::PopID();
}

void spotLightsSection(std::vector<scene::SpotLight>& lights) {
	if (!ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}
	ImGui::PushID("spots");
	if (ImGui::Button("Add spot light")) {
		lights.push_back({});
	}
	for (size_t i = 0; i < lights.size(); ++i) {
		scene::SpotLight& l = lights[i];
		ImGui::PushID(int(i));
		ImGui::SeparatorText(("Spot light " + std::to_string(i + 1)).c_str());
		ImGui::DragFloat3("Position", &l.position.x, 0.02f);
		direction(l.yaw_degrees, l.pitch_degrees);
		colorAndIntensity(l.color, l.intensity);
		ImGui::DragFloat("Inner angle", &l.inner_degrees, 0.2f, 0.0f, 80.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
		ImGui::DragFloat("Outer angle", &l.outer_degrees, 0.2f, 0.0f, 80.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
		l.outer_degrees = std::max(l.outer_degrees, l.inner_degrees); // внешний угол не меньше внутреннего
		const bool remove = ImGui::Button("Remove");
		ImGui::PopID();
		if (remove) {
			lights.erase(lights.begin() + i);
			break;
		}
	}
	ImGui::PopID();
}

} // namespace

void drawSceneWindow(scene::Scene& s) {
	ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400.0f, 650.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 4: shadow mapping");
	ImGui::PushItemWidth(200.0f);

	if (ImGui::CollapsingHeader("Shadow mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat("Sun coverage width", &s.sun_shadow_width, 2.0f, 40.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::Text("Sun coverage: %.1f x %.1f", s.sun_shadow_width, s.sun_shadow_width);
		ImGui::SliderFloat("Depth bias", &s.shadow_bias, 0.0f, 8.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::TextDisabled(s.spot_lights.empty() ? "No spot light" : "Spot light 1 casts a shadow");
	}

	cameraSection(s.camera);
	objectsSection(s);

	ImGui::SeparatorText("Selected pyramid");
	scene::Object& object = s.objects[s.selected];
	ImGui::PushID(s.selected);
	transformSection(object);
	motionSection(object.motion);
	materialSection(object);
	ImGui::PopID();

	ImGui::SeparatorText("Lighting");
	ImGui::Checkbox("Show light sources", &s.show_lights);
	ambientAndDirectionalSection(s);
	pointLightsSection(s.point_lights);
	spotLightsSection(s.spot_lights);

	ImGui::PopItemWidth();
	ImGui::End();
}

} // namespace ui
