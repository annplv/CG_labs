#include "geometry.hpp"

namespace geometry {

Mesh makePyramid() {
	const glm::vec3 points[5] = {
		{ -0.5f, -0.5f, -0.5f }, // основание сзади слева
		{  0.5f, -0.5f, -0.5f }, // основание сзади справа
		{  0.5f, -0.5f,  0.5f }, // основание спереди справа
		{ -0.5f, -0.5f,  0.5f }, // основание спереди слева
		{  0.0f,  0.5f,  0.0f }, // вершина
	};

	Mesh mesh;
	for (const glm::vec3& p : points) {
		const glm::vec3 srgb = p + 0.5f;  // координаты в диапазоне [-0.5, 0.5], добавляем 0.5 для [0,1]
		mesh.vertices.push_back({ p, glm::pow(srgb, glm::vec3(2.2f)) });  // степень 2.2 для перевода цвета в линейное пр-во
	}

	mesh.indices = {
		0, 1, 2,  0, 2, 3, // основание (2 треуг)
		3, 2, 4,           // передняя грань 
		2, 1, 4,           // правая 
		1, 0, 4,           // задняя 
		0, 3, 4,           // левая
	};
	return mesh;
}

} // namespace geometry
