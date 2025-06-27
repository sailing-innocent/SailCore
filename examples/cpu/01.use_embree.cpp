/**
  * @file 00.pure.cpp
  * @brief Embree Demo - Pure C++ Code
  * @author sailing-innocent
  * @date 2025-02-10
  */

#include <embree4/rtcore.h>
#include <limits>
#include <iostream>

int main() {
	RTCDevice device = rtcNewDevice(nullptr);
	RTCScene scene = rtcNewScene(device);
	RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	float* vb = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), 3);
	vb[0] = -1.0f;
	vb[1] = -1.0f;
	vb[2] = 0.0f;// vertex 0
	vb[3] = 1.0f;
	vb[4] = -1.0f;
	vb[5] = 0.0f;// vertex 1
	vb[6] = 0.0f;
	vb[7] = 1.0f;
	vb[8] = 0.0f;// vertex 2

	unsigned* ib = (unsigned*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 1);
	ib[0] = 0;
	ib[1] = 1;
	ib[2] = 2;

	rtcCommitGeometry(geom);
	rtcAttachGeometry(scene, geom);
	rtcReleaseGeometry(geom);
	rtcCommitScene(scene);

	RTCRayHit rayhit;
	rayhit.ray.org_x = 0.0f;// Ray origin
	rayhit.ray.org_y = 0.0f;
	rayhit.ray.org_z = -1.0f;
	rayhit.ray.dir_x = 0.0f;// Ray direction
	rayhit.ray.dir_y = 0.0f;
	rayhit.ray.dir_z = 1.0f;// Shooting towards +Z
	rayhit.ray.tnear = 0.0f;
	rayhit.ray.tfar = std::numeric_limits<float>::infinity();
	rayhit.ray.mask = -1;
	rayhit.ray.flags = 0;
	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	rtcIntersect1(scene, &rayhit, &args);

	// hit triangle ID
	std::cout << "hit.geomID = " << rayhit.hit.geomID << std::endl;
	std::cout << "hit.primID = " << rayhit.hit.primID << std::endl;
	// normal vector (not normalized) of the hit point
	std::cout << "hit.Ng_x = " << rayhit.hit.Ng_x << std::endl;
	std::cout << "hit.Ng_y = " << rayhit.hit.Ng_y << std::endl;
	std::cout << "hit.Ng_z = " << rayhit.hit.Ng_z << std::endl;
	// triangle barycentric coordinates of the hit point on hit triangle
	std::cout << "hit.u = " << rayhit.hit.u << std::endl;
	std::cout << "hit.v = " << rayhit.hit.v << std::endl;

	if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
		std::cout << "Intersection at t = " << rayhit.ray.tfar << std::endl;
	} else {
		std::cout << "No Intersection" << std::endl;
	}

	rtcReleaseScene(scene);
	rtcReleaseDevice(device);
	return 0;
}