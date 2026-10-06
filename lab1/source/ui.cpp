#include "ui.hpp"

#include <cstdio>

#include <imgui.h>

namespace ui {

	namespace {

		void cameraSection(scene::Camera& camera) {
			if (!ImGui::CollapsingHeader("Camera & projection", ImGuiTreeNodeFlags_DefaultOpen)) {
				return;
			}
			if (ImGui::RadioButton("Perspective", camera.projection == scene::Projection::Perspective)) {
				camera.projection = scene::Projection::Perspective;
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("Orthographic", camera.projection == scene::Projection::Orthographic)) {
				camera.projection = scene::Projection::Orthographic;
			}
		}

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

		void colorSection(scene::Object& o) {
			if (!ImGui::CollapsingHeader("Color", ImGuiTreeNodeFlags_DefaultOpen)) {
				return;
			}
			ImGui::ColorEdit3("Color", &o.color.x);
			ImGui::Checkbox("Vertex colors", &o.vertex_colors);
		}

	} // namespace

	void drawSceneWindow(scene::Scene& s) {
		ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
		ImGui::Begin("Lab 1: pyramid", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::PushItemWidth(220.0f);

		cameraSection(s.camera);
		objectsSection(s);

		ImGui::SeparatorText("Selected pyramid");
		scene::Object& object = s.objects[s.selected];
		transformSection(object);
		motionSection(object.motion);
		colorSection(object);

		ImGui::PopItemWidth();
		ImGui::End();
	}

} // namespace ui