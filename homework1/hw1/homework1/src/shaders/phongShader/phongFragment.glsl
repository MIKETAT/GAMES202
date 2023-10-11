#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586
#define BLOCKER_FILTER_SIZE 0.02
#define LIGHT_WEIGHT 0.1

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
  // 查找遮挡物的平均深度，通过相似计算出filterSize，用于PCSS第三步的采样
  float totalDepth = 0.;
  int numOfBlocker = 0;
  for (int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i++) {
    vec2 sample_coord = uv + poissonDisk[i] * BLOCKER_FILTER_SIZE;
    float sampleDepth = unpack(texture2D(shadowMap, sample_coord));
    if (sampleDepth + EPS < zReceiver) {
      totalDepth = totalDepth + sampleDepth;
      numOfBlocker++;
    }
  }
  if (numOfBlocker == 0) {
    return 1.;
  }
	return totalDepth / float(numOfBlocker);
}

// 软阴影/阴影抗锯齿
float PCF(sampler2D shadowMap, vec4 coords) {
  // 当前深度
  float currentDepth = coords.z;

  /*
  texture2d(shadowMap, coords)  coords的坐标范围是 [0, 1] 这是自然，纹理坐标嘛 
  因此这里要做的采样就是在 投影空间坐标 -> shadowMap上的纹理坐标 + 随机的偏置
  重复一定次数，求出平均的visibility
  那我们的偏置应当是多少呢，已知纹理坐标范围是[0, 1], poissonDisk[i]的范围也是[0, 1]
  我们需要的就是一个filterSize, 这个filterSize * poissonDisk[i] + coords 就是最终在shadowMap上采样的坐标
  网上的做法是，定义 stride 和 shadowMapSize, 我的理解没错的话,
  filterSize = stride / shadowMapSize, 也就是 (10, 100) 与 (1, 10） 无异
  // 更新: 但是 stride / shadowMapSize 的写法更能感受filterSize的大小，例如 16 / 1024， 在 1024*1024的纹理上采样周围16*16个点
  */
  
  //PCF的逻辑,采样ShadowMap上的若干点,得到深度与当前深度比较,得到[0, 1]的visibility
  float visibility = 0.;
  //float stride = 32.;
  //float shadowMapSize = 2048.;
  float filterSize = 16. / 1024.;

  for (int i = 0; i < PCF_NUM_SAMPLES; i++) {
    // 本次采样坐标
    vec2 sampleCoords = poissonDisk[i] * filterSize + coords.xy;

    // 获取本次采样得到的深度值
    float shadowDepth = unpack(texture2D(shadowMap, sampleCoords));
  
    // 统计本次的visibility, 加上bias
    if (currentDepth < shadowDepth + EPS) {
      visibility = visibility + 1.;
    }
  }
  return visibility / float(PCF_NUM_SAMPLES);
}


float PCSS(sampler2D shadowMap, vec4 coords) {
  float filterSize = 32. / 2048.;
  float zReceiver = coords.z;
  
  // STEP 1: avgblocker depth
  float blockerDepth = findBlocker(shadowMap, coords.xy, zReceiver);
  
  // STEP 2: penumbra size
  float W_Pebumbra = float(LIGHT_WEIGHT) * (zReceiver - blockerDepth) / blockerDepth;
  
  // STEP 3: filtering
  float visibility = 0.;
  for (int i = 0; i < PCF_NUM_SAMPLES; i++) {
    // 本次采样坐标
    vec2 sampleCoords = poissonDisk[i] * filterSize * W_Pebumbra + coords.xy;

    // 获取本次采样得到的深度值
    float shadowDepth = unpack(texture2D(shadowMap, sampleCoords));
  
    // 统计本次的visibility, 加上bias
    if (zReceiver < shadowDepth + EPS) {
      visibility = visibility + 1.;
    }
  }
  return visibility / float(PCF_NUM_SAMPLES);
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  float shadowDepth = unpack(texture2D(shadowMap, shadowCoord.xy));
  float currentDepth = shadowCoord.z;
  // add bias to solve self occulusion
  if (currentDepth > shadowDepth + EPS) {
    return 0.0;
  }
  return 1.0;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {
  // 归一化， vPositionFromLight 是经过 light MVP 变化后的坐标, (light MVP 正交投影后变换到 [-1, 1]^3 空间中)
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  
  // [-1, 1]^3 -> [0, 1]^3
  shadowCoord.xyz = (shadowCoord.xyz + 1.0) / 2.0;

  // 生成 poissonDisk
  //vec2 randomSeed = vec2(rand_2to1(shadowCoord.xy), rand_2to1(shadowCoord.xy));
  poissonDiskSamples(shadowCoord.xy);

  float visibility;
  //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  //visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //  gl_FragColor = vec4(phongColor, 1.0);
}