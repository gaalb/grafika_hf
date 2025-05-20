//=============================================================================================
// Zöld háromszög: pucér, lábbal hajtós megoldás
//=============================================================================================
#include "../framework.h"

// csúcspont árnyaló
const char * vertSource = R"(
	#version 330				
    precision highp float;

	layout(location = 0) in vec2 cP;	// 0. bemeneti regiszter

	void main() {
		gl_Position = vec4(cP.x, cP.y, 0, 1); 	// bemenet már normalizált eszközkoordinátákban
	}
)";

// pixel árnyaló
const char * fragSource = R"(
	#version 330
    precision highp float;

	uniform vec3 color;			// konstans szín
	out vec4 fragmentColor;		// pixel szín

	void main() {
		fragmentColor = vec4(color, 1); // RGB -> RGBA
	}
)";

const int winWidth = 600, winHeight = 600;

class GreenTriangleApp : public glApp {
	unsigned int vao;   // geometria
	unsigned int prog;	// csúcspont és pixel árnyalók
public:
	GreenTriangleApp() : glApp("Green triangle") { }

	// Inicializáció, 
	void onInitialization() {
		glGenVertexArrays(1, &vao);	// egy vao azonosító kérése
		glBindVertexArray(vao);		// a kapott vao-t aktiváljuk
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);		// egy vbo azonosító kérése
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // a kapott vbo-t aktiváljuk

		// Geometria 24 bájttal (6 float vagy 3x2 koordináta)
		float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };
		glBufferData(GL_ARRAY_BUFFER, 	// VBO GPU-ra másolása
			         sizeof(vertices), vertices,	// másolt bájtok száma és címe
			         GL_STATIC_DRAW);	// nem fogjuk módosítani

		glEnableVertexAttribArray(0);  	// 0. regiszter engedélyezése
		glVertexAttribPointer(0,   		// vbo -> 0. regiszter
			                  2, GL_FLOAT, GL_FALSE, // 2 float/nem fix pontos
			                  0, NULL); 	// tömören pakolt

 		// Csúcspont árnyaló: azonosító generálás, forrás feltöltés, fordítás 
		unsigned int vertShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertShader, 1, &vertSource, NULL);
		glCompileShader(vertShader);

		// Pixel árnyaló: azonosító generálás, forrás feltöltés, fordítás 
		unsigned int fragShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragShader, 1, &fragSource, NULL);
		glCompileShader(fragShader);

		// Program = csúcspont-pixelárnyaló pár létrehozása
		prog = glCreateProgram();
		glAttachShader(prog, vertShader);
		glAttachShader(prog, fragShader);
		glLinkProgram(prog); // a csúcspont-pixelárnyaló pár szerkesztése
		glUseProgram(prog);  // ez fusson
	}

	// Ablak újrarajzolás
	void onDisplay() {
		glClearColor(0, 0, 0, 0);     // háttér szín
		glClear(GL_COLOR_BUFFER_BIT); // rasztertár törlés
		glViewport(0, 0, winWidth, winHeight);

		int location = glGetUniformLocation(prog, "color");
		glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 float, zöld szín

		glBindVertexArray(vao);  // Rajzolási hívás
		glDrawArrays(GL_TRIANGLES, 0 /* start index */, 3 /* elem szám */);
	}
} app;

