#ifndef MATERIAL_H
#define MATERIAL_H

#include "framework.h"
#include "shader.h"
#include "camera.h"
#include "mesh.h"
#include "extra/hdre.h"

#include "scene_data.h"

extern sSceneData scene_data;

class Material {
public:

	Shader* shader = NULL;
	Texture* texture = NULL;
	vec4 color;

	virtual void setUniforms(Camera* camera, Matrix44 model) = 0;
	virtual void render(Mesh* mesh, Matrix44 model, Camera * camera) = 0;
	virtual void renderInMenu() = 0;
};

class StandardMaterial : public Material {
public:

	StandardMaterial();
	~StandardMaterial();

	void setUniforms(Camera* camera, Matrix44 model);
	void render(Mesh* mesh, Matrix44 model, Camera * camera);
	void renderInMenu();
};

class WireframeMaterial : public StandardMaterial {
public:

	WireframeMaterial();
	~WireframeMaterial();

	void render(Mesh* mesh, Matrix44 model, Camera * camera);
};


// =================
// CUSTOM MATERIALS
// =================

class TexturedMaterial : public StandardMaterial {
public:
	TexturedMaterial(const char* texture_name);
	~TexturedMaterial();

	void renderInMenu();
};

class PhongMaterial : public StandardMaterial {
public:
	float ambient_value = 0.5f;
	float diffuse_value = 0.6f;
	float specular_value = 0.5f;
	float shiniess = 32.0f;

	Vector4 light_color = {1.0f, 1.0f, 1.0f, 1.0f};
	Vector3 light_position = {-14.0f, -14.0f, 14.0f};

	PhongMaterial();
	~PhongMaterial();

	void setUniforms(Camera* camera, Matrix44 model);
	void renderInMenu();
};

class SkyBoxMaterial : public StandardMaterial {
public:
	SkyBoxMaterial();
	~SkyBoxMaterial();

	void setCubemapTexture(const char* texture);

	void renderInMenu();
};

class ReflectiveMaterial : public StandardMaterial {
public:
	float reflectiveness = 0.6f;

	ReflectiveMaterial();

	void setUniforms(Camera* camera, Matrix44 model);
	void renderInMenu();
};

class HDReMaterial : public StandardMaterial {
public:
	HDRE* curr_hdre = NULL;
	int display_level = 0;
	Texture* enviorment = NULL;
	Texture* prem[5] = { NULL };

	HDReMaterial();
	~HDReMaterial();

	void setHDReTexture(const char* dir);
	void setUniforms(Camera* camera, Matrix44 model);
};


enum ePBR_Format : int {
	SEPARETE_TEXTURES = 0,
	G_ROUGH_B_METAL = 1
};

class PBRMaterial : public StandardMaterial {
public:
	Texture* albedo_map = NULL;
	Texture* roughness_map = NULL;
	Texture* metalness_map = NULL;
	Texture* normal_map = NULL;
	Texture* brdf_LUT = NULL;

	HDReMaterial* hdr_material = NULL;

	int render_output = 0;

	ePBR_Format texture_mode = SEPARETE_TEXTURES;

	PBRMaterial(const char*         albedo_dir, 
				const char*         roughness_dir, 
				const char*         metalness_dir,
				const ePBR_Format   texture_mode);

	~PBRMaterial();

	void setUniforms(Camera* camera, Matrix44 model);
	void renderInMenu();
};
#endif