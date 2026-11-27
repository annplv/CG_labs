#include "geometry.hpp"

#include <cmath>
#include <numbers>

namespace geometry {

namespace {

// uvs — текстурные координаты для каждого элемента triangles (по одной на вершину треугольника);
// если пусто, координаты нулевые (кубу источников текстура не нужна)
Mesh buildFlat(const std::vector<glm::vec3>& points, const std::vector<uint16_t>& triangles,
               const std::vector<glm::vec2>& uvs = {}) {
	Mesh mesh;
	for (size_t i = 0; i < triangles.size(); i += 3) {
		const glm::vec3 a = points[triangles[i]];
		const glm::vec3 b = points[triangles[i + 1]];
		const glm::vec3 c = points[triangles[i + 2]];
		const glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));

		for (size_t k = 0; k < 3; ++k) {
			const glm::vec2 uv = uvs.empty() ? glm::vec2(0.0f) : uvs[i + k];
			mesh.indices.push_back(uint16_t(mesh.vertices.size()));
			mesh.vertices.push_back({ points[triangles[i + k]], n, uv });
		}
	}
	return mesh;
}

} // namespace

Mesh makePyramid() {
	const std::vector<glm::vec3> points = {
		{ -0.5f, -0.5f, -0.5f }, // основание сзади слева
		{  0.5f, -0.5f, -0.5f }, // основание сзади справа
		{  0.5f, -0.5f,  0.5f }, // основание спереди справа
		{ -0.5f, -0.5f,  0.5f }, // основание спереди слева
		{  0.0f,  0.5f,  0.0f }, // вершина
	};

	
	auto base = [&](uint16_t i) { return glm::vec2(points[i].x + 0.5f, 0.5f - points[i].z); };
	const std::vector<glm::vec2> side = { { 0.0f, 1.0f }, { 1.0f, 1.0f }, { 0.5f, 0.0f } };

	std::vector<glm::vec2> uvs = { base(0), base(1), base(2), base(0), base(2), base(3) };
	for (int face = 0; face < 4; ++face) {
		uvs.insert(uvs.end(), side.begin(), side.end());
	}

	return buildFlat(points, {
		0, 1, 2,  0, 2, 3, // основание
		3, 2, 4,           // передняя грань
		2, 1, 4,           // правая
		1, 0, 4,           // задняя
		0, 3, 4,           // левая
	}, uvs);
}

// Два треугольника в плоскости XZ. При стандартной анимации пирамида всегда выше пола.
Mesh makeFloor() {
	return buildFlat({ {-7, -1.5f, -7}, {7, -1.5f, -7}, {7, -1.5f, 7}, {-7, -1.5f, 7} },
	                 { 0, 3, 2, 0, 2, 1 });
}

Mesh makeCube() {
	std::vector<glm::vec3> points;
	for (int i = 0; i < 8; ++i) {
		points.push_back({ (i & 1) ? 0.5f : -0.5f, (i & 2) ? 0.5f : -0.5f, (i & 4) ? 0.5f : -0.5f });
	}
	return buildFlat(points, {
		0, 1, 3,  0, 3, 2, // z = -0.5
		4, 6, 7,  4, 7, 5, // z = +0.5
		0, 2, 6,  0, 6, 4, // x = -0.5
		1, 5, 7,  1, 7, 3, // x = +0.5
		0, 4, 5,  0, 5, 1, // y = -0.5
		2, 3, 7,  2, 7, 6, // y = +0.5
	});
}

Mesh makeSpotCone() {
	constexpr int segments = 32;
	Mesh mesh;
	mesh.vertices.push_back({ glm::vec3(0.0f), glm::vec3(0.0f), glm::vec2(0.0f) }); // 0 — вершина конуса
	mesh.vertices.push_back({ glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f), glm::vec2(0.0f) }); // 1 — центр основания
	for (int k = 0; k < segments; ++k) {
		const float a = 2.0f * std::numbers::pi_v<float> * float(k) / float(segments);
		mesh.vertices.push_back({ glm::vec3(std::cos(a), std::sin(a), 1.0f), glm::vec3(0.0f), glm::vec2(0.0f) });
	}

	mesh.indices = { 0, 1 }; // ось
	for (int k = 0; k < segments; ++k) {
		const uint16_t current = uint16_t(2 + k);
		const uint16_t next = uint16_t(2 + (k + 1) % segments);
		mesh.indices.insert(mesh.indices.end(), { current, next }); // окружность основания
		if (k % 4 == 0) {
			mesh.indices.insert(mesh.indices.end(), { 0, current }); // образующая
		}
	}
	return mesh;
}

} // namespace geometry
