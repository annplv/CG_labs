#include "geometry.hpp"

namespace geometry {

namespace {

Mesh buildFlat(const std::vector<glm::vec3>& points, const std::vector<uint16_t>& triangles) {
	Mesh mesh;
	for (size_t i = 0; i < triangles.size(); i += 3) {
		const glm::vec3 a = points[triangles[i]];
		const glm::vec3 b = points[triangles[i + 1]];
		const glm::vec3 c = points[triangles[i + 2]];
		const glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));

		for (const glm::vec3& p : { a, b, c }) {
			const glm::vec3 color = glm::pow(p + 0.5f, glm::vec3(2.2f));
			mesh.indices.push_back(uint16_t(mesh.vertices.size()));
			mesh.vertices.push_back({ p, n, color });
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
	
	return buildFlat(points, {
		0, 1, 2,  0, 2, 3, // основание
		3, 2, 4,           // передняя грань
		2, 1, 4,           // правая
		1, 0, 4,           // задняя
		0, 3, 4,           // левая
	});
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

} // namespace geometry
