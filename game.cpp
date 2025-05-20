#include "framework.h"
#include <algorithm>
#include <iostream>
#include "triangleIntersects.h"

static const bool live = true;
static bool gameOver = false;

using threeyd::moeller::TriangleIntersects;
using std::cout;
using std::endl;
static const float startingRotationSpeed = 0.2f;
static float rotationSpeed = startingRotationSpeed;
static const float startingSpawnInterval = 15.0f * M_PI / 180.f;
static float spawnInterval = startingSpawnInterval;
static const float doublingTime = 15.0f;
static float elapsedTime = 0.0f;
static float gameOverTime = 0.0f;
static const float restartDelay = 2.0f;

static const float eps = 0.0001;
static const float cylinderWidth = 100;
static const float flightSpace = 10;
static const float flightRadius = 50;
static const float spawnAngle = -70 * M_PI / 180.0f;
static const float despawnAngle = 5.0f * M_PI / 180.0f;
static const int checkerBoardSize = 100;

//---------------------------
struct Material {
	//---------------------------
	vec3 kd = vec3(1, 1, 1), ks = vec3(0, 0, 0), ka = vec3(1, 1, 1); // sorrendben diffúz, spekuláris, ambiens
	float shininess = 1, emission = 0;
};

const char* phongVertexSource = R"(
		#version 330
		precision highp float;
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		uniform mat4 MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3 wEye;
		
		layout(location=0) in vec3 vtxPos;
		layout(location=1) in vec3 vtxNorm;
		layout(location=2) in vec2 vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec2 texcoord;
		
		void main() { 
			gl_Position = MVP * vec4(vtxPos, 1);
			vec4 wPos = M * vec4(vtxPos, 1); 
			for (int i=0; i<nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;	
			}
			wView = wEye * wPos.w - wPos.xyz;
			wNormal = (vec4(vtxNorm, 0) * Minv).xyz;
			texcoord = vtxUV;
		}		
	)";

const char* phongFragmentSource = R"(
		#version 330
		precision highp float;
		
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess, emission;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int nLights;
		uniform sampler2D diffuseTexture; //?? TODO

		in vec3 wNormal;
		in vec3 wView;
		in vec3 wLight[8];
		in vec2 texcoord;
		
		out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);	
			//if (dot(N, V) < 0) N = -N; // lusta voltam 2 cylindert csinálni, ezért ez offos itt
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;
			vec3 emission = material.emission * texColor;

			vec3 radiance = emission;
			for (int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L+V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N, H), 0);
				radiance += ka * lights[i].La + 
						(kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
				
			}
			fragmentColor = vec4(radiance, 1);
		}		
	)";

void rotationToAngleAndAxis(vec3 original, vec3 rotated, vec3& rotationAxis, float& rotationAngle) {
	rotationAxis = normalize(cross(original, rotated));
	rotationAngle = acosf(dot(original, rotated) / length(original) / length(rotated));
}

// egyenletes eloszlású véletlen szám, -1 és 1 között
inline float PMRND() { return 2.0f * rand() / RAND_MAX - 1.0f; }

// egyenletes eloszlású véletlen szám, mean+var és mean-var között
inline float Rand(float mean, float var) { return mean + PMRND() * var; }

const int winWidth = 1000, winHeight = 600;

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) { return Dnum(f * r.f, f * r.d + d * r.f); }
	Dnum operator/(Dnum r) { return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f); }
};

template<class T> Dnum<T> Exp(Dnum<T>g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return Dnum<T>(powf(g.f), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos; //homogén, végtelenben: direkcionális fényforrás
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	        MVP, M, Minv, V, P;
	Material* material = nullptr;
	std::vector<Light> lights;
	Texture* texture = nullptr;
	vec3	        wEye;
};

Texture* uniformTexture;
Texture* yellowAndBlueTexture;
Texture* whiteAndBlackTexture;
Material* spaceShipMaterial;
Material* cylinderMaterial;
Material* obstacleMaterial;
Material* hitMaterial;

//---------------------------
class PhongShader : public GPUProgram {
	//---------------------------
public:
	PhongShader() : GPUProgram(phongVertexSource, phongFragmentSource) {}

	void setUniformMaterial(const Material* material, const std::string& name) {
		setUniform(material->kd, name + ".kd");
		setUniform(material->ka, name + ".ka");
		setUniform(material->kd, name + ".kd");
		setUniform(material->ks, name + ".ks");
		setUniform(material->shininess, name + ".shininess");
		setUniform(material->emission, name + ".emission");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.P * state.V * state.M, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		int textureUnit = 0; // textúra mintavevõ egység
		setUniform(textureUnit, "diffuseTexture");
		if (state.texture != nullptr) state.texture->Bind(textureUnit);
		static Material defaultMaterial;
		const Material* material = (state.material == nullptr) ? &defaultMaterial : state.material;
		setUniformMaterial(material, "material");
		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
struct VtxData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Mesh : public Geometry<VtxData> {
	//---------------------------
public:
	virtual void Draw() = 0;
	Mesh() {
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VtxData), (void*)offsetof(VtxData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VtxData), (void*)offsetof(VtxData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VtxData), (void*)offsetof(VtxData, texcoord));
	}
};

//---------------------------
class OBJSurface : public Mesh {
	//---------------------------
public:
	OBJSurface(std::string pathname, float scale) {
		std::vector<vec3> vertices, normals;
		std::vector<vec2> uvs;
		std::ifstream read;
		char line[256];
		read.open(pathname);
		if (!read.is_open()) {
			printf("%s cannot be opened\n", pathname.c_str());
		}
		while (!read.eof()) {
			read.getline(line, 256);
			float x, y, z;
			if (sscanf(line, "v %f %f %f\n", &x, &y, &z) == 3) {
				vertices.push_back(vec3(x * scale, y * scale, z * scale));
				continue;
			}
			if (sscanf(line, "vn %f %f %f\n", &x, &y, &z) == 3) {
				normals.push_back(vec3(x, y, z));
				continue;
			}
			if (sscanf(line, "vt %f %f\n", &x, &y) == 2) {
				uvs.push_back(vec2(x, y));
				continue;
			}
			int v[4], t[4], n[4];
			VtxData vd[4];
			if (sscanf(line, "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n",
				&v[0], &t[0], &n[0], &v[1], &t[1], &n[1], &v[2], &t[2], &n[2], &v[3], &t[3], &n[3]) == 12) {
				for (int i = 0; i < 4; ++i) {
					vd[i].position = vertices[v[i] - 1]; vd[i].texcoord = uvs[t[i] - 1]; vd[i].normal = normals[n[i] - 1];
				}
				vtx.push_back(vd[0]); vtx.push_back(vd[1]); vtx.push_back(vd[2]);
				vtx.push_back(vd[0]); vtx.push_back(vd[2]); vtx.push_back(vd[3]);
			}
			if (sscanf(line, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
				&v[0], &t[0], &n[0], &v[1], &t[1], &n[1], &v[2], &t[2], &n[2]) == 9) {
				for (int i = 0; i < 3; ++i) {
					vd[i].position = vertices[v[i] - 1]; vd[i].texcoord = uvs[t[i] - 1]; vd[i].normal = normals[n[i] - 1];
					vtx.push_back(vd[i]);
				}
			}
		}
		read.close();
		updateGPU();
	}
	void Draw() {
		Bind();
		glDrawArrays(GL_TRIANGLES, 0, vtx.size());
	}
};

class ManualMesh : public Mesh {
public:
	void addTriangle(vec3 v1, vec3 v2, vec3 v3) {
		//feltételezve van a jobbkéz körüljárás, a textúra egyelõre el van engedve
		VtxData vtxData[3];
		vec3 normal = normalize(cross(v2 - v1, v3 - v2));
		vtxData[0].position = v1;
		vtxData[0].normal = normal;
		vtxData[0].texcoord = vec2(0, 0);
		vtxData[1].position = v2;
		vtxData[1].normal = normal;
		vtxData[1].texcoord = vec2(0, 0);
		vtxData[2].position = v3;
		vtxData[2].normal = normal;
		vtxData[2].texcoord = vec2(0, 0);
		vtx.push_back(vtxData[0]);
		vtx.push_back(vtxData[1]);
		vtx.push_back(vtxData[2]);
		updateGPU();
	}
	void Draw() {
		Bind();
		glDrawArrays(GL_TRIANGLES, 0, vtx.size());
	}
};

//---------------------------
class ParamSurface : public Mesh {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	virtual VtxData GenVtxData(float u, float v) {
		VtxData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1)); // du/du=1, du/dv=0, dv/du=0, dv/dv=1
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = 40, int M = 40) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtx.push_back(GenVtxData((float)j / M, (float)i / N));
				vtx.push_back(GenVtxData((float)j / M, (float)(i + 1) / N));
			}
		}
		updateGPU();
	}
	void Draw() {
		Bind();
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		Dnum2 _U = U * 2.0f * (float)M_PI;
		Dnum2 _V = V * (float)M_PI;
		X = Cos(_U) * Sin(_V);
		Y = Sin(_U) * Sin(_V);
		Z = Cos(_V);
	}
};

class Cylinder : public ParamSurface {
	bool invertedNormal;
public:
	Cylinder(bool _invertedNormal = true) {
		invertedNormal = _invertedNormal;
		create();
	}
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		Dnum2 _U = U * 2.0f * (float)M_PI;
		Dnum2 _V = V * 2.0f - 1.0f;
		X = invertedNormal ? _V * -1.0f : _V;
		Y = Cos(_U);
		Z = Sin(_U);
	}
};

class Scene;
const vec3 nullvec = vec3(0, 0, 0);

//---------------------------
struct GameObject {
	//---------------------------
	vec3   position = nullvec, velocity = nullvec, acceleration = nullvec;
	bool   alive = true;
	std::vector<GameObject*> children;

	virtual void Control(float tstart, float tend, Scene* scene) {} // állapotszámítás
	virtual void Animate(float tstart, float tend) {        // állapot váltás
		float dt = tend - tstart;
		position += velocity * dt;          // Euler integrálás
		velocity += acceleration * dt;
	}
	virtual void Draw(RenderState state) {}  // rajzolás

	virtual void Kill() { alive = false; }
};

//---------------------------
struct MeshGameObject : GameObject {
	//---------------------------
	static PhongShader* shader;
	Texture* texture = nullptr;
	Material* material = nullptr;
	Mesh* geometry = nullptr;
	vec3 size, rotationAxis;
	float rotationAngle;

	void setPose(vec3 pos, vec3 rotAxis, float rotAngle) {
		position = pos;
		rotationAxis = rotAxis;
		rotationAngle = rotAngle;
	}

	virtual void setModelingTransform(mat4& M, mat4& Minv) {
		M = translate(position) * rotate(rotationAngle, rotationAxis) * scale(size);
		Minv = scale(vec3(1 / size.x, 1 / size.y, 1 / size.z)) * rotate(-rotationAngle, rotationAxis) * translate(-position);
	}
	virtual void Draw(RenderState state) {
		mat4 M, Minv;
		setModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.P * state.V * state.M;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
		for (auto child : children) child->Draw(state);
	}
};

class ColumnObstacle : public MeshGameObject {
	float xPos;
	void build() {
		std::vector<vec3> points;
		points.push_back(vec3(-0.5, -0.5, 0.5)); //0
		points.push_back(vec3(0.5, -0.5, 0.5)); //1
		points.push_back(vec3(0.5, -0.5, -0.5)); //2
		points.push_back(vec3(-0.5, -0.5, -0.5)); //3
		points.push_back(vec3(-0.5, 0.5, 0.5)); //4
		points.push_back(vec3(0.5, 0.5, 0.5)); //5
		points.push_back(vec3(0.5, 0.5, -0.5)); //6
		points.push_back(vec3(-0.5, 0.5, -0.5)); //7

		ManualMesh* mesh = new ManualMesh();
		mesh->addTriangle(points[1], points[0], points[3]);
		mesh->addTriangle(points[3], points[2], points[1]);
		mesh->addTriangle(points[4], points[0], points[1]);
		mesh->addTriangle(points[1], points[5], points[4]);
		mesh->addTriangle(points[0], points[4], points[3]);
		mesh->addTriangle(points[4], points[7], points[3]);
		mesh->addTriangle(points[7], points[2], points[3]);
		mesh->addTriangle(points[7], points[6], points[2]);
		mesh->addTriangle(points[6], points[5], points[2]);
		mesh->addTriangle(points[5], points[1], points[2]);
		mesh->addTriangle(points[7], points[5], points[6]);
		mesh->addTriangle(points[7], points[4], points[5]);
		geometry = mesh;
	}
public:
	ColumnObstacle(float _xPos, vec3 _size,
		Texture* _texture, Material* _material) {
		xPos = _xPos;
		texture = _texture;
		material = _material;
		size = _size;
		vec3 pos = vec3(rotate(spawnAngle, vec3(1, 0, 0)) * vec4(xPos, flightRadius, 0, 1));
		pos.y -= flightRadius;
		setPose(pos, vec3(1, 0, 0), spawnAngle);
		build();
	}
	void Animate(float tstart, float tend) {
		float dt = tend - tstart;
		rotationAngle += rotationSpeed * dt;
		position = vec3(rotate(rotationAngle, vec3(1, 0, 0)) * vec4(xPos, flightRadius, 0, 1));
		position.y -= flightRadius;
	}
	void Control(float tstart, float tend, Scene* scene) {
		if (!alive) return;
		if (rotationAngle > despawnAngle) {
			Kill();
		}
	}
};

struct Camera { // 3D kamera
public:
	vec3 wEye, wLookat, wVup; // külsõ paraméterek
	float fov, asp, fp, bp; // belsõ paraméterek
	Camera() {
		asp = (float)winWidth / winHeight; // aspektus arány
		fov = 90.0f * (float)M_PI / 180.0f; // függõleges látószög
		fp = 0.1f; bp = 200.0f; // elsõ és hátsó vágósík távolság
	}
	// Nézeti transzformáció
	mat4 V() { return lookAt(wEye, wLookat, wVup); }
	// Perspektív transzformáció
	mat4 P() { return perspective(fov, asp, fp, bp); }
};

//--------------------------------------------
struct Avatar : public MeshGameObject {
	//--------------------------------------------
	Camera camera;

	void setCamera(vec3 wEye, vec3 wLookat, vec3 wVup) {
		camera.wEye = wEye;
		camera.wLookat = wLookat;
		camera.wVup = wVup;
	}

	virtual void ProcessInput(Scene* scene) = 0;

	mat4 V() { return camera.V(); }
	mat4 P() { return camera.P(); }
};

//---------------------------
class Scene {
	//---------------------------
	int current = 0;
	std::vector<GameObject*> objects[2]; // ping-pong
	Avatar* avatar;
	std::vector<Light> lights;
	float spawnAngleCounter = 0;
public:
	void Build();

	void Render() {
		RenderState state;
		state.wEye = avatar->position;
		state.V = avatar->V();
		state.P = avatar->P();
		state.lights = lights;
		// First pass: opaque objects
		for (auto* obj : objects[current]) obj->Draw(state);
		// Second pass: transparent objects
		// VolumetricGameObject::Flush(state);
	}
	void spawnObstacle() {
		float x = Rand(0.0f, cylinderWidth / 2 - 1);
		float sizeX = Rand(2.0f, 1.0f);
		float sizeZ = Rand(1.0f, 0.5f);
		ColumnObstacle* obs = new ColumnObstacle(x, vec3(sizeX, flightSpace, sizeZ), uniformTexture, obstacleMaterial);
		Join(obs);
	}

	std::vector<GameObject*>& Objects() { return objects[current]; }
	void Simulate(float tstart, float tend) {
		avatar->ProcessInput(this);
		const float dt = 0.05f; // dt kicsi (20 FPS)
		for (float t = tstart; t < tend; t += dt) {
			float Dt = fmin(dt, tend - t);
			for (auto* obj : objects[current]) obj->Control(t, t + Dt, this); // control
			for (auto* obj : objects[current]) {
				if (obj->alive) Join(obj);
				else            delete obj; // bury dead
			}
			objects[current].clear();
			current = 1 - current;
			for (auto* obj : objects[current]) obj->Animate(t, t + Dt); // animate

			spawnAngleCounter += rotationSpeed * dt;
			if (spawnAngleCounter > spawnInterval) {
				spawnAngleCounter = 0;
				spawnObstacle();
			}
		}
	}
	void Join(GameObject* obj) { objects[!current].push_back(obj); }

	void clear() {
		for (GameObject* obj : Objects()) {
			if (dynamic_cast<ColumnObstacle*>(obj)) {
				obj->Kill();
			}
		}
	}
	void restart() {
		clear();
		elapsedTime = 0.0f;
		rotationSpeed = startingRotationSpeed;
		spawnInterval = startingSpawnInterval;
	}
};


class SpaceShip : public Avatar {
	void build() {
		std::vector<vec3> points;
		points.push_back(vec3(0.5, -0.5, 0)); //0
		points.push_back(vec3(-0.5, -0.5, 0)); //1
		points.push_back(vec3(-1, 0, 0)); //2
		points.push_back(vec3(-0.5, 0.5, 0)); //3
		points.push_back(vec3(0.5, 0.5, 0)); //4
		points.push_back(vec3(1, 0, 0)); // 5
		points.push_back(vec3(0.25, 0, -0.5)); // 6
		points.push_back(vec3(0, 0, -1)); // 7
		points.push_back(vec3(-0.25, 0, -0.5)); // 8

		ManualMesh* mesh = new ManualMesh();
		mesh->addTriangle(points[0], points[4], points[1]);
		mesh->addTriangle(points[4], points[3], points[1]);
		mesh->addTriangle(points[5], points[4], points[0]);
		mesh->addTriangle(points[1], points[3], points[2]);

		mesh->addTriangle(points[0], points[7], points[1]);
		mesh->addTriangle(points[7], points[4], points[3]);
		mesh->addTriangle(points[5], points[6], points[0]);
		mesh->addTriangle(points[6], points[7], points[0]);
		mesh->addTriangle(points[5], points[4], points[6]);
		mesh->addTriangle(points[7], points[6], points[4]);
		mesh->addTriangle(points[2], points[1], points[8]);
		mesh->addTriangle(points[3], points[2], points[8]);
		mesh->addTriangle(points[8], points[1], points[7]);
		mesh->addTriangle(points[3], points[8], points[7]);
		geometry = mesh;
	}
	float tiltSpeed;
	static const float tilt2Vel, maxTilt, defaultTiltSpeed;
	bool intersects(ColumnObstacle* obst) {
		// TODO: itt a háromszögek számolása majdnem tuti hogy rossz, mert
		// az elforgatatlan és skálázatlan verziók között van ütközés számolva
		float wObst = obst->size.x / 2;
		float hObst = obst->size.z / 2;
		float w = size.x;
		float h = size.z;
		//if (obst->position.z > obst->size.z / 2) return false;
		vec3 dr = position - obst->position;
		if (fabs(dr.x) > wObst + w) return false;
		if (fabs(dr.z) > hObst + h) return false;
		for (int i = 0; i < geometry->Vtx().size(); i += 3) {
			mat4 M, Minv, obstM, obstMinv;
			setModelingTransform(M, Minv);			
			vec3 a0 = vec3(M * vec4(geometry->Vtx()[i].position, 1));
			vec3 a1 = vec3(M * vec4(geometry->Vtx()[i + 1].position, 1));
			vec3 a2 = vec3(M * vec4(geometry->Vtx()[i + 2].position, 1));
			for (int j = 0; j < obst->geometry->Vtx().size(); j += 3) {
				setModelingTransform(obstM, obstMinv);
				vec3 b0 = vec3(obstM * vec4(obst->geometry->Vtx()[j].position, 1));
				vec3 b1 = vec3(obstM * vec4(obst->geometry->Vtx()[j+1].position, 1));
				vec3 b2 = vec3(obstM * vec4(obst->geometry->Vtx()[j+2].position, 1));
				if (TriangleIntersects<vec3>::triangle(a0, a1, a2, b0, b1, b2)) {
					return true;
				}
			}
		}

	}
public:
	SpaceShip(vec3 position, vec3 cameraPos, vec3 lookat, vec3 wVup,
		Texture* playerTexture, Material* playerMaterial,
		vec3 playerSize, vec3 playerRotationAxis) {
		setCamera(cameraPos, lookat, wVup);
		setPose(position, playerRotationAxis, 0);
		texture = playerTexture;
		material = playerMaterial;
		size = playerSize;
		tiltSpeed = 0;
		build();
	}
	void Control(float tstart, float tend, Scene* scene) {
		material->kd = vec3(0.6f, 0.6f, 0.6f);
		if (elapsedTime < restartDelay) return; // disable collisions at startup		
		for (auto* obj : scene->Objects()) {
			if (dynamic_cast<ColumnObstacle*>(obj)) {
				ColumnObstacle* obst = (ColumnObstacle*)obj;
				if (intersects(obst)) {
					material->kd = vec3(2.0f, 0, 0);
					obst->material =hitMaterial;
					if (live) {
						gameOver = true;						
						gameOverTime = elapsedTime;
						cout << "YOUR TIME: " << elapsedTime << " seconds" << endl;
					}
				}
			}
		}
	}

	void Animate(float tstart, float tend) {
		float dt = tend - tstart;
		position += velocity * dt;          // Euler integrálás
		//velocity += acceleration * dt;
		rotationAngle += tiltSpeed * dt;
		if (rotationAngle >= maxTilt) {
			rotationAngle = maxTilt;
		}
		if (rotationAngle <= -maxTilt) {
			rotationAngle = -maxTilt;
		}
		velocity.x = -rotationAngle * tilt2Vel;
		camera.wEye.x = position.x;
		camera.wLookat.x = position.x;
	}

	void ProcessInput(Scene* scene) {
		bool left = pollKey(KEY_LEFT);
		bool right = pollKey(KEY_RIGHT);
		if (rotationAngle > eps) tiltSpeed = -defaultTiltSpeed; // return to level
		if (rotationAngle < -eps) tiltSpeed = defaultTiltSpeed; // return to level
		if (left && !right) tiltSpeed = defaultTiltSpeed; // input overwrites return to level
		if (right && !left) tiltSpeed = -defaultTiltSpeed; // input overwrites return to level
		if (position.x > 40) tiltSpeed = defaultTiltSpeed; // map constraint overwrites input
		if (position.x < -40) tiltSpeed = -defaultTiltSpeed; // map constraint overwrites input
	}
	void Kill() {

	}
};

const float SpaceShip::tilt2Vel = 15.0f / (45 * M_PI / 180);
const float SpaceShip::maxTilt = 70.0f * M_PI / 180;
const float SpaceShip::defaultTiltSpeed = SpaceShip::maxTilt / 0.4f;

class RotatingCylinder : public MeshGameObject {
public:
	RotatingCylinder(vec3 pos, vec3 _size, vec3 _rotationAxis, Texture* _texture,
		Material* _material) {
		size = _size;
		texture = _texture;
		material = _material;
		setPose(pos, _rotationAxis, 0);
	}
	void Animate(float tstart, float tend) {
		float dt = tend - tstart;
		geometry = new Cylinder();
		rotationAngle += rotationSpeed * dt;
	}
};

PhongShader* MeshGameObject::shader;

//--------------------------------------------
void Scene::Build() {
	//--------------------------------------------
	vec3 yellow(1, 1, 0);
	vec3 blue(0, 0, 1);
	vec3 white(0.8, 0.8, 0.8);
	vec3 black(0, 0, 0);
	std::vector<vec3> color(1);
	color[0] = white;
	uniformTexture = new Texture(1, 1, color);

	std::vector<vec3> yellowAndBlueImage(checkerBoardSize * checkerBoardSize);
	std::vector<vec3> blackAndWhiteImage(checkerBoardSize * checkerBoardSize);
	for (int x = 0; x < checkerBoardSize; x++) for (int y = 0; y < checkerBoardSize; y++) {
		yellowAndBlueImage[y * checkerBoardSize + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		blackAndWhiteImage[y * checkerBoardSize + x] = (x & 1) ^ (y & 1) ? white : black;
	}
	yellowAndBlueTexture = new Texture(checkerBoardSize, checkerBoardSize, yellowAndBlueImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	whiteAndBlackTexture = new Texture(checkerBoardSize, checkerBoardSize, blackAndWhiteImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	spaceShipMaterial = new Material();
	spaceShipMaterial->ka = vec3(0.9f, 0, 0.9f);
	spaceShipMaterial->kd = vec3(1.2f, 0, 1.2f);
	spaceShipMaterial->ks = vec3(3, 3, 3);
	spaceShipMaterial->shininess = 50;


	cylinderMaterial = new Material();
	cylinderMaterial->ka = vec3(0.6f, 0.6f, 0.6f);
	cylinderMaterial->kd = vec3(1, 1, 1);
	cylinderMaterial->ks = vec3(0, 0, 0);
	cylinderMaterial->shininess = 1;

	obstacleMaterial = new Material();
	obstacleMaterial->ka = vec3(0, 1.0f, 0);
	obstacleMaterial->kd = vec3(0, 2.0f, 0);

	hitMaterial = new Material();
	hitMaterial->ka = vec3(1.0f, 0, 0);
	hitMaterial->kd = vec3(2.0f, 0, 0);

	vec3 playerPos(0, 0, 0);
	vec3 cameraPos(0, 3.75, 2);
	vec3 cameraLookat(0, 3, 0);
	vec3 wVup(0, 1, 0);
	vec3 playerSize = vec3(1, 0.5, 2) * 0.6f;
	vec3 playerAxis(0, 0, 1);
	avatar = new SpaceShip(playerPos, cameraPos, cameraLookat, wVup, uniformTexture, spaceShipMaterial, playerSize, playerAxis);
	Join(avatar);

	vec3 cylinderCenter(0, -flightRadius, 0);
	vec3 outerSize(cylinderWidth, flightRadius + flightSpace / 2, flightRadius + flightSpace / 2);
	vec3 innerSize(cylinderWidth, flightRadius - flightSpace / 2, flightRadius - flightSpace / 2);
	vec3 cylinderAxis(1, 0, 0);

	RotatingCylinder* outerCylinder = new RotatingCylinder(cylinderCenter, outerSize, cylinderAxis, yellowAndBlueTexture, cylinderMaterial);
	RotatingCylinder* innerCylinder = new RotatingCylinder(cylinderCenter, innerSize, cylinderAxis, whiteAndBlackTexture, cylinderMaterial);
	Join(outerCylinder);
	Join(innerCylinder);

	lights.resize(2);
	lights[0].La = vec3(0.2f, 0.2f, 0.2f); lights[0].Le = vec3(0.8f, 0.8f, 0.8f); lights[0].wLightPos = vec4(0.2f, -1.0f, -0.5f, 0);
	lights[1].La = vec3(0, 0, 0); lights[1].Le = vec3(0.4f, 0.4f, 0.4f); lights[1].wLightPos = vec4(0, 1.0f, 1.0f, 0);
};

class GameApp : public glApp {
	Scene* scene;
public:
	GameApp() : glApp(3, 3, winWidth, winHeight, "CRQEYD NHF") {}
	void onInitialization() {
		glClearColor(0, 0, 0, 0);
		// statikus árnyalók létrehozása
		MeshGameObject::shader = new PhongShader();
		glEnable(GL_DEPTH_TEST); // mélységpuffer algoritmus bekapcsolása
		glDisable(GL_CULL_FACE); // hátsólap eldobás tiltása
		scene = new Scene;
		scene->Build();
	}
	void onDisplay() {
		glViewport(0, 0, winWidth, winHeight);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		scene->Render();
	}
	void onTimeElapsed(float tstart, float tend) {
		float dt = tend - tstart;
		elapsedTime += dt;
		if (!gameOver) {
			scene->Simulate(tstart, tend);			
			rotationSpeed += dt * startingRotationSpeed / doublingTime;
			float spawnRate = 1.0f / spawnInterval;
			spawnRate += dt * startingSpawnInterval / doublingTime;
			spawnInterval = 1.0f / spawnRate;			
		}
		else {
			if (elapsedTime > gameOverTime + restartDelay) {
				gameOver = false;
				scene->restart();
			}
		}
		refreshScreen();
	}

	virtual void onKeyboard(int key) {
	}
} app;
