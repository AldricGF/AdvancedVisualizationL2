varying vec3 v_world_position;
varying vec3 v_normal;
varying vec2 v_uv;

uniform vec4 u_light_radiance;
uniform vec3 u_light_position;
uniform vec3 u_camera_position;

// Material maps
uniform sampler2D u_roughness_map;
uniform sampler2D u_metal_map;
uniform sampler2D u_albedo_map;
uniform sampler2D u_normal_map;

uniform sampler2D u_brdf_LUT;


// HDRE textures
uniform samplerCube u_texture_enviorment; 
uniform samplerCube u_texture_prem_0; 
uniform samplerCube u_texture_prem_1; 
uniform samplerCube u_texture_prem_2; 
uniform samplerCube u_texture_prem_3; 
uniform samplerCube u_texture_prem_4; 

uniform float u_output_mode;
uniform float u_material_mode;
// TODO: add the other maps

#define PI 3.14159265359
#define RECIPROCAL_PI 0.3183098861837697

const float GAMMA = 2.2;
const float INV_GAMMA = 1.0 / GAMMA;

struct sVectors {
    vec3 normal;
    vec3 view;
    vec3 light;
    vec3 reflect;
    vec3 half_v;
    vec3 tangent_view;
    float n_dot_v;
    float n_dot_h;
    float l_dot_n;
};

struct sMaterial {
    float roughness;
    float metalness;
    
    vec3 base_color;
    vec3 diffuse_color;
    vec3 specular_color;

    float alpha;
};

// PROVIDED FUNCTIONS ===============
vec3 getReflectionColor(vec3 r, float roughness)
{
	float lod = roughness * 5.0;

	vec4 color;

	if(lod < 1.0) color = mix( textureCube(u_texture_enviorment, r), textureCube(u_texture_prem_0, r), lod );
	else if(lod < 2.0) color = mix( textureCube(u_texture_prem_0, r), textureCube(u_texture_prem_1, r), lod - 1.0 );
	else if(lod < 3.0) color = mix( textureCube(u_texture_prem_1, r), textureCube(u_texture_prem_2, r), lod - 2.0 );
	else if(lod < 4.0) color = mix( textureCube(u_texture_prem_2, r), textureCube(u_texture_prem_3, r), lod - 3.0 );
	else if(lod < 5.0) color = mix( textureCube(u_texture_prem_3, r), textureCube(u_texture_prem_4, r), lod - 4.0 );
	else color = textureCube(u_texture_prem_4, r);

	return color.rgb;
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// degamma
vec3 gamma_to_linear(vec3 color)
{
	return pow(color, vec3(GAMMA));
}

// gamma
vec3 linear_to_gamma(vec3 color)
{
	return pow(color, vec3(INV_GAMMA));
}

//cotangent_frame was provided
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv){
	// get edge vectors of the pixel triangle
	vec3 dp1 = dFdx( p );
	vec3 dp2 = dFdy( p );
	vec2 duv1 = dFdx( uv );
	vec2 duv2 = dFdy( uv );

	// solve the linear system
	vec3 dp2perp = cross( dp2, N );
	vec3 dp1perp = cross( N, dp1 );
	vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
	vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

	// construct a scale-invariant frame
	float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
	return mat3( T * invmax, B * invmax, N );
}

//perturbNormal was provided, but we commented some lines
vec3 perturbNormal( vec3 N, vec3 V, vec2 texcoord, vec3 normal_pixel ){
	//#ifdef USE_POINTS //changed: commented
	//return N; //chagned: commented
	//#endif //changed: commented

	// assume N, the interpolated vertex normal and
	// V, the view vector (vertex to eye)
	//vec3 normal_pixel = texture2D(normalmap, texcoord ).xyz;
	normal_pixel = normal_pixel * 255./127. - 128./127.;
	mat3 TBN = cotangent_frame(N, V, texcoord);
	return normalize(TBN * normal_pixel);
}

// CUSTOM FUNCTIONS ===============

//Direct Lighting
vec3 directBRDF(sMaterial material, sVectors vects){

	vec3 c_diff = mix(vec3(0.0), material.diffuse_color, material.roughness);
	float F0 = mix(vec3(0.04), material.base_color, material.metalness);
	
	//f_lambert
	vec3 f_lambert = c_diff * RECIPROCAL_PI;
	
	//f_pfacet: F
	vec3 F = fresnelSchlick(vects.n_dot_v, F0);
		
	//f_pfacet: G
	float k = (1.0 + material.roughness)*(1.0 + material.roughness) / 8;
	float G1 = vects.n_dot_v / (vects.n_dot_v * (1.0 - k) + k);
	float G2 = vects.l_dot_n / (vects.l_dot_n * (1.0 - k) + k);
	float G = G1 * G2;

	//f_pfacet: D
	float alpha = material.roughness * material.roughness;
    float alpha_squared = alpha * alpha;
	float denom = ((vects.n_dot_h * vects.n_dot_h) * (alpha_squared - 1.0)) + 1.0;
	float D = alpha_squared / (PI * denom * denom);
	
	//f_pfacet: all
	vec3 f_pfacet = (F * G * D) / (4.0 * vects.n_dot_v * vects.l_dot_n); //+0.001?
	
	//Total BRDF
	return (f_lambert + f_pfacet) * vects.l_dot_n;
}

sVectors computeVectors() {
    sVectors result;
    result.normal = normalize(v_normal);
    result.view = normalize(u_camera_position - v_world_position);
    result.light = normalize(u_light_position - v_world_position);
    result.half_v = normalize(result.view + result.light);
    result.reflect = normalize(reflect(-result.view, result.normal));

	//Detailed Normals
	vec3 normal = texture2D(u_normal_map, v_uv);
	result.normal = perturbNormal(result.normal, result.view, v_uv, normal);

    // Clamped dots
    result.n_dot_v = clamp(dot(result.normal, result.view), 0.001, 1.0);
    result.n_dot_h = clamp(dot(result.normal, result.half_v), 0.001, 1.0);
    result.l_dot_n = clamp(dot(result.normal, result.light), 0.001, 1.0);

    // From world to tangent space
    mat3 inv_TBN = transpose(cotangent_frame(result.normal, result.view, v_uv));
    result.tangent_view = inv_TBN * result.view;

    return result;
}

//////////////////////////////////////////////////////////////////////////////////////////

void main() {
	sMaterial frag_material;
	sVectors frag_vectors = computeVectors(); 

    vec3 world_position = v_world_position;
    vec3 light_pos = u_light_position;

    //Initialize albedo, roughness, and metalness
	//Albedo
	vec4 alb_color = texture2D(u_albedo_map, v_uv);
	//Albedo: Gamma to linear
    frag_material.base_color = gamma_to_linear(alb_color.rgb);
    frag_material.diffuse_color = frag_material.base_color;
    frag_material.alpha = alb_color.a;
	//Roughness
	frag_material.roughness = texture2D(u_roughness_map, v_uv).g;
	//Metalness
	frag_material.metalness = texture2D(u_roughness_map, v_uv).b;

    //Calculate Direct Lighting
    vec3 output_color = directBRDF(frag_material, frag_vectors);

    //Correct gamma
    output_color = linear_to_gamma(output_color);

    //Return this
    gl_FragColor = vec4(output_color, 1.0);
}