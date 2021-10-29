#pragma once
#include "scenenode.h"
#include "material.h"
#include "texture.h"

static const char* SKYBOXES[3] = { 
	"data/environments/city",
	"data/environments/dragonvale",
	"data/environments/snow"
};

static const char* HDREs[5] = {
	"data/environments/panorama.hdre",
	"data/environments/pisa.hdre",
	"data/environments/studio.hdre",
	"data/environments/san_giuseppe_bridge.hdre",
	"data/environments/tv_studio.hdre"
};

class SkyboxNode : public SceneNode {
public:	
	int enviorment_id = 0;

	SkyboxNode();
	~SkyboxNode();
	
	void render(Camera* camera);
	void renderInMenu();
};

class HDReSkyboxNode : public SceneNode {
public:
	int hdre_id = 0;
	int represented_blur = 0;

	HDReSkyboxNode();
	~HDReSkyboxNode();

	void render(Camera* camera);
	void renderInMenu();
};