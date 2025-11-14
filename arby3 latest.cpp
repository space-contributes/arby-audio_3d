// Arby Ultimate - CV-Enhanced Spatial Audio Engine with Automatic Source Detection
// Consolidated linear algebra + motion tracking + original audio processing
// Zero external dependencies (OpenCV functionality built-in)

#pragma once

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <array>
#include <fstream>
#include <string>
#include <map>
#include <queue>

// ============================================================================
// CONFIGURATION
// ============================================================================

namespace Config {
constexpr int SAMPLE_RATE = 96000;
constexpr int CHUNK_SIZE = 512;
constexpr float SPEED_OF_SOUND = 343.0f;
constexpr int MAX_DELAY_SAMPLES = 96000 * 2;
constexpr int MAX_REFLECTION_ORDER = 3;
constexpr int MAX_SPEAKERS = 32;
constexpr int NUM_OCTAVE_BANDS = 7;

// CV Configuration
constexpr int MAX_TRACKED_OBJECTS = 16;
constexpr int MOTION_HISTORY_FRAMES = 30;
constexpr float MOTION_THRESHOLD = 0.15f;
constexpr int CV_WIDTH = 320;
constexpr int CV_HEIGHT = 240;
}

// ============================================================================
// VECTOR MATH
// ============================================================================

struct Vec3 {
float x, y, z;
Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
float length() const { return std::sqrt(x*x + y*y + z*z); }
Vec3 normalize() const {
float l = length();
return l > 1e-6f ? Vec3(x/l, y/l, z/l) : Vec3(0, 0, 0);
}
};

struct Vec2 {
float x, y;
Vec2(float x = 0, float y = 0) : x(x), y(y) {}
Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
float length() const { return std::sqrt(x*x + y*y); }
};

// ============================================================================
// LINEAR ALGEBRA CONSOLIDATION
// All spatial audio processing unified into matrix operations
// ============================================================================

class SpatialMatrix {
public:
// Consolidated transfer function H(ω, θ, φ, d, r)
// Where: ω=frequency, θ=azimuth, φ=elevation, d=distance, r=room_params
static float calculateTransferCoefficient(
float frequency,
float azimuth, // degrees
float elevation, // degrees
float distance,
float roomAbsorption,
int reflectionOrder) {

// 1. Distance attenuation: 1/d with non-linear scaling
float distGain = 1.0f / (1.0f + 0.5f * std::pow(distance, 1.2f));

// 2. Air absorption (ISO 9613-1 simplified)
float airLoss = std::exp(-frequency * distance / 50000.0f);

// 3. Reflection loss (exponential decay per order)
float reflectionLoss = std::pow(1.0f - roomAbsorption, reflectionOrder);

// 4. Frequency-dependent directivity
float directivity = 1.0f + 0.3f * std::cos(azimuth * 3.14159f / 180.0f) *
(1.0f - std::abs(frequency - 1000.0f) / 10000.0f);

// Combined transfer function
return distGain * airLoss * reflectionLoss * directivity;
}

// VBAP (Vector Base Amplitude Panning) - consolidated speaker gain calculation
static std::vector<float> calculateVBAP(
float azimuth,
float elevation,
const std::vector<Vec3>& speakerPositions) {

std::vector<float> gains(speakerPositions.size(), 0.0f);

float azRad = azimuth * 3.14159f / 180.0f;
float elRad = elevation * 3.14159f / 180.0f;

Vec3 sourceDir(
std::cos(elRad) * std::sin(azRad),
std::sin(elRad),
std::cos(elRad) * std::cos(azRad)
);

float sumSquared = 0.0f;
for (size_t i = 0; i < speakerPositions.size(); i++) {
Vec3 spkDir = speakerPositions[i].normalize();
float cosAngle = std::max(0.0f, sourceDir.dot(spkDir));
gains[i] = cosAngle * cosAngle; // Energy-preserving
sumSquared += gains[i] * gains[i];
}

// Normalize for constant power
float rms = std::sqrt(sumSquared);
if (rms > 1e-6f) {
for (auto& g : gains) g /= rms;
}

return gains;
}
};

// ============================================================================
// BIQUAD FILTER
// ============================================================================

class BiquadFilter {
private:
float b0, b1, b2, a1, a2;
float z1, z2;

public:
BiquadFilter() : b0(1), b1(0), b2(0), a1(0), a2(0), z1(0), z2(0) {}

void setLowPass(float freq, float Q, float sampleRate) {
float w0 = 2.0f * 3.14159265f * freq / sampleRate;
float cosw0 = std::cos(w0);
float sinw0 = std::sin(w0);
float alpha = sinw0 / (2.0f * Q);
float a0 = 1.0f + alpha;
b0 = ((1.0f - cosw0) / 2.0f) / a0;
b1 = (1.0f - cosw0) / a0;
b2 = ((1.0f - cosw0) / 2.0f) / a0;
a1 = (-2.0f * cosw0) / a0;
a2 = (1.0f - alpha) / a0;
}

void setBandPass(float freq, float Q, float sampleRate) {
float w0 = 2.0f * 3.14159265f * freq / sampleRate;
float cosw0 = std::cos(w0);
float sinw0 = std::sin(w0);
float alpha = sinw0 / (2.0f * Q);
float a0 = 1.0f + alpha;
b0 = alpha / a0;
b1 = 0.0f;
b2 = -alpha / a0;
a1 = (-2.0f * cosw0) / a0;
a2 = (1.0f - alpha) / a0;
}

float process(float input) {
float output = b0 * input + z1;
z1 = b1 * input - a1 * output + z2;
z2 = b2 * input - a2 * output;
return output;
}

void reset() { z1 = z2 = 0; }
};

// ============================================================================
// DELAY LINE
// ============================================================================

class DelayLine {
private:
std::vector<float> buffer;
int writePos;
int size;

public:
DelayLine(int maxDelay = Config::MAX_DELAY_SAMPLES)
: buffer(maxDelay + 1, 0.0f), writePos(0), size(maxDelay + 1) {}

void write(float sample) {
buffer[writePos] = sample;
writePos = (writePos + 1) % size;
}

float read(float delaySamples) const {
delaySamples = std::clamp(delaySamples, 0.0f, (float)(size - 2));
int delayInt = (int)delaySamples;
float frac = delaySamples - delayInt;
int readPos = (writePos - delayInt - 1 + size) % size;
int readPos2 = (readPos - 1 + size) % size;
return buffer[readPos] * (1.0f - frac) + buffer[readPos2] * frac;
}

void clear() {
std::fill(buffer.begin(), buffer.end(), 0.0f);
writePos = 0;
}
};

// ============================================================================
// BUILT-IN COMPUTER VISION (NO DEPENDENCIES)
// ============================================================================

struct BoundingBox {
int x, y, width, height;
float confidence;
};

struct TrackedObject {
BoundingBox box;
Vec2 velocity;
std::vector<Vec2> positionHistory;
float audioCorrelation;
int trackId;
int framesTracked;
bool isActive;
};

class SimpleFrameBuffer {
private:
std::vector<uint8_t> data;
int width, height;

public:
SimpleFrameBuffer(int w = Config::CV_WIDTH, int h = Config::CV_HEIGHT)
: width(w), height(h), data(w * h, 0) {}

void setPixel(int x, int y, uint8_t value) {
if (x >= 0 && x < width && y >= 0 && y < height) {
data[y * width + x] = value;
}
}

uint8_t getPixel(int x, int y) const {
if (x >= 0 && x < width && y >= 0 && y < height) {
return data[y * width + x];
}
return 0;
}

void fill(uint8_t value) {
std::fill(data.begin(), data.end(), value);
}

void copyFrom(const uint8_t* src, int srcWidth, int srcHeight, int channels) {
// Downsample and convert to grayscale if needed
for (int y = 0; y < height; y++) {
for (int x = 0; x < width; x++) {
int srcX = x * srcWidth / width;
int srcY = y * srcHeight / height;
int srcIdx = (srcY * srcWidth + srcX) * channels;

if (channels == 3 || channels == 4) {
// RGB/RGBA to grayscale
data[y * width + x] = (uint8_t)(
0.299f * src[srcIdx] +
0.587f * src[srcIdx + 1] +
0.114f * src[srcIdx + 2]
);
} else {
data[y * width + x] = src[srcIdx];
}
}
}
}

int getWidth() const { return width; }
int getHeight() const { return height; }
const uint8_t* getData() const { return data.data(); }
};

class MotionDetector {
private:
SimpleFrameBuffer previousFrame;
SimpleFrameBuffer currentFrame;
SimpleFrameBuffer diffFrame;
std::vector<TrackedObject> trackedObjects;
int nextTrackId;
std::mutex cvMutex;

public:
MotionDetector() : nextTrackId(0) {}

void processFrame(const uint8_t* frameData, int width, int height, int channels) {
std::lock_guard<std::mutex> lock(cvMutex);

previousFrame = currentFrame;
currentFrame.copyFrom(frameData, width, height, channels);

if (previousFrame.getData()[0] == 0) {
return; // First frame, skip
}

// Calculate frame difference
computeDifference();

// Detect motion regions
auto motionRegions = detectMotionRegions();

// Update tracked objects
updateTracking(motionRegions);
}

const std::vector<TrackedObject>& getTrackedObjects() const {
return trackedObjects;
}

Vec2 estimatePrimaryMotionDirection() const {
Vec2 avgVelocity(0, 0);
int activeCount = 0;

for (const auto& obj : trackedObjects) {
if (obj.isActive && obj.framesTracked > 3) {
avgVelocity = avgVelocity + obj.velocity;
activeCount++;
}
}

if (activeCount > 0) {
avgVelocity = avgVelocity * (1.0f / activeCount);
}

return avgVelocity;
}

private:
void computeDifference() {
int width = currentFrame.getWidth();
int height = currentFrame.getHeight();

for (int y = 0; y < height; y++) {
for (int x = 0; x < width; x++) {
int diff = std::abs((int)currentFrame.getPixel(x, y) -
(int)previousFrame.getPixel(x, y));
diffFrame.setPixel(x, y, (uint8_t)std::min(diff, 255));
}
}
}

std::vector<BoundingBox> detectMotionRegions() {
std::vector<BoundingBox> regions;
int width = diffFrame.getWidth();
int height = diffFrame.getHeight();

// Simple blob detection with grid-based approach
const int gridSize = 16;
std::vector<std::vector<float>> grid(
height / gridSize + 1,
std::vector<float>(width / gridSize + 1, 0.0f)
);

// Accumulate motion in grid cells
for (int y = 0; y < height; y++) {
for (int x = 0; x < width; x++) {
float motionValue = diffFrame.getPixel(x, y) / 255.0f;
if (motionValue > Config::MOTION_THRESHOLD) {
grid[y / gridSize][x / gridSize] += motionValue;
}
}
}

// Find significant motion clusters
for (int gy = 0; gy < (int)grid.size(); gy++) {
for (int gx = 0; gx < (int)grid[0].size(); gx++) {
if (grid[gy][gx] > gridSize * gridSize * Config::MOTION_THRESHOLD) {
BoundingBox box;
box.x = gx * gridSize;
box.y = gy * gridSize;
box.width = gridSize;
box.height = gridSize;
box.confidence = grid[gy][gx] / (gridSize * gridSize);
regions.push_back(box);
}
}
}

return regions;
}

void updateTracking(const std::vector<BoundingBox>& detections) {
// Mark all as inactive
for (auto& obj : trackedObjects) {
obj.isActive = false;
}

// Match detections to existing tracks
for (const auto& detection : detections) {
Vec2 detectionCenter(
detection.x + detection.width / 2.0f,
detection.y + detection.height / 2.0f
);

bool matched = false;
float minDist = 50.0f; // Max matching distance
int bestMatch = -1;

for (size_t i = 0; i < trackedObjects.size(); i++) {
if (!trackedObjects[i].isActive &&
trackedObjects[i].positionHistory.size() > 0) {

Vec2 lastPos = trackedObjects[i].positionHistory.back();
float dist = (detectionCenter - lastPos).length();

if (dist < minDist) {
minDist = dist;
bestMatch = i;
matched = true;
}
}
}

if (matched && bestMatch >= 0) {
// Update existing track
auto& obj = trackedObjects[bestMatch];
obj.box = detection;
obj.isActive = true;
obj.framesTracked++;

Vec2 newPos = detectionCenter;
if (obj.positionHistory.size() > 0) {
Vec2 lastPos = obj.positionHistory.back();
obj.velocity = newPos - lastPos;
}

obj.positionHistory.push_back(newPos);
if (obj.positionHistory.size() > Config::MOTION_HISTORY_FRAMES) {
obj.positionHistory.erase(obj.positionHistory.begin());
}
} else {
// Create new track
if (trackedObjects.size() < Config::MAX_TRACKED_OBJECTS) {
TrackedObject newObj;
newObj.box = detection;
newObj.velocity = Vec2(0, 0);
newObj.positionHistory.push_back(detectionCenter);
newObj.audioCorrelation = 0.0f;
newObj.trackId = nextTrackId++;
newObj.framesTracked = 1;
newObj.isActive = true;
trackedObjects.push_back(newObj);
}
}
}

// Remove stale tracks
trackedObjects.erase(
std::remove_if(trackedObjects.begin(), trackedObjects.end(),
[](const TrackedObject& obj) {
return !obj.isActive && obj.framesTracked > 10;
}),
trackedObjects.end()
);
}
};

// ============================================================================
// SURROUND SOUND DETECTION
// ============================================================================

class SurroundDetector {
private:
std::vector<float> channelEnergyHistory[8]; // Track up to 7.1 channels
int historySize = 100;
std::mutex detectorMutex;
bool isSurroundDetected = false;
bool isStereoUpmixed = false;
bool isMonoUpscaled = false;

public:
void analyzeChannels(const std::vector<std::vector<float>>& channelData) {
std::lock_guard<std::mutex> lock(detectorMutex);

if (channelData.size() < 2) return;

// Calculate energy per channel
std::vector<float> channelEnergies(channelData.size(), 0.0f);
for (size_t ch = 0; ch < channelData.size(); ch++) {
for (float sample : channelData[ch]) {
channelEnergies[ch] += sample * sample;
}
channelEnergies[ch] /= channelData[ch].size();
}

// Store history
for (size_t ch = 0; ch < std::min(channelData.size(), (size_t)8); ch++) {
channelEnergyHistory[ch].push_back(channelEnergies[ch]);
if (channelEnergyHistory[ch].size() > historySize) {
channelEnergyHistory[ch].erase(channelEnergyHistory[ch].begin());
}
}

// Detect patterns
detectSurroundPattern();
}

bool isTrueSurround() const { return isSurroundDetected && !isStereoUpmixed && !isMonoUpscaled; }
bool isFakeSurround() const { return isStereoUpmixed || isMonoUpscaled; }
bool shouldEnableCV() const { return !isTrueSurround(); }

private:
void detectSurroundPattern() {
if (channelEnergyHistory[0].size() < historySize / 2) return;

// Calculate correlation between channels
std::vector<std::vector<float>> correlations(8, std::vector<float>(8, 0.0f));

for (int ch1 = 0; ch1 < 8; ch1++) {
for (int ch2 = ch1 + 1; ch2 < 8; ch2++) {
if (channelEnergyHistory[ch1].empty() || channelEnergyHistory[ch2].empty()) continue;

float corr = calculateCorrelation(
channelEnergyHistory[ch1],
channelEnergyHistory[ch2]
);
correlations[ch1][ch2] = corr;
}
}

// Detect mono upscaling (all channels highly correlated)
float avgCorrelation = 0.0f;
int corrCount = 0;
for (int i = 0; i < 8; i++) {
for (int j = i + 1; j < 8; j++) {
avgCorrelation += correlations[i][j];
corrCount++;
}
}
if (corrCount > 0) avgCorrelation /= corrCount;

isMonoUpscaled = (avgCorrelation > 0.95f);

// Detect stereo upmix (L/R correlated, others derived)
float lrCorrelation = correlations[0][1];
isStereoUpmixed = (lrCorrelation > 0.85f && avgCorrelation > 0.7f && !isMonoUpscaled);

// Detect true surround (low inter-channel correlation, significant energy differences)
float energyVariance = 0.0f;
float avgEnergy = 0.0f;
for (int ch = 0; ch < 8; ch++) {
if (!channelEnergyHistory[ch].empty()) {
float chAvg = 0.0f;
for (float e : channelEnergyHistory[ch]) chAvg += e;
chAvg /= channelEnergyHistory[ch].size();
avgEnergy += chAvg;
}
}
avgEnergy /= 8.0f;

for (int ch = 0; ch < 8; ch++) {
if (!channelEnergyHistory[ch].empty()) {
float chAvg = 0.0f;
for (float e : channelEnergyHistory[ch]) chAvg += e;
chAvg /= channelEnergyHistory[ch].size();
energyVariance += (chAvg - avgEnergy) * (chAvg - avgEnergy);
}
}
energyVariance /= 8.0f;

isSurroundDetected = (avgCorrelation < 0.6f && energyVariance > 0.01f);
}

float calculateCorrelation(const std::vector<float>& a, const std::vector<float>& b) {
if (a.size() != b.size() || a.empty()) return 0.0f;

float meanA = 0.0f, meanB = 0.0f;
for (size_t i = 0; i < a.size(); i++) {
meanA += a[i];
meanB += b[i];
}
meanA /= a.size();
meanB /= b.size();

float numerator = 0.0f, denomA = 0.0f, denomB = 0.0f;
for (size_t i = 0; i < a.size(); i++) {
float da = a[i] - meanA;
float db = b[i] - meanB;
numerator += da * db;
denomA += da * da;
denomB += db * db;
}

float denom = std::sqrt(denomA * denomB);
return (denom > 1e-6f) ? (numerator / denom) : 0.0f;
}
};

// ============================================================================
// AUDIO-MOTION CORRELATOR
// ============================================================================

class AudioMotionCorrelator {
private:
std::vector<float> audioEnergyHistory;
std::vector<float> motionEnergyHistory;
int historySize = 50;

public:
float correlate(float audioEnergy, float motionEnergy) {
audioEnergyHistory.push_back(audioEnergy);
motionEnergyHistory.push_back(motionEnergy);

if (audioEnergyHistory.size() > historySize) {
audioEnergyHistory.erase(audioEnergyHistory.begin());
motionEnergyHistory.erase(motionEnergyHistory.begin());
}

if (audioEnergyHistory.size() < 10) return 0.0f;

// Calculate cross-correlation with lag
float maxCorr = 0.0f;
for (int lag = -5; lag <= 5; lag++) {
float corr = 0.0f;
int count = 0;

for (int i = std::max(0, -lag);
i < (int)audioEnergyHistory.size() + std::min(0, -lag);
i++) {
int audioIdx = i;
int motionIdx = i + lag;

if (motionIdx >= 0 && motionIdx < (int)motionEnergyHistory.size()) {
corr += audioEnergyHistory[audioIdx] * motionEnergyHistory[motionIdx];
count++;
}
}

if (count > 0) corr /= count;
maxCorr = std::max(maxCorr, corr);
}

return maxCorr;
}
};

// ============================================================================
// SPEAKER CONFIGURATION
// ============================================================================

enum class SpeakerType {
LEFT, RIGHT, CENTER, LFE,
SURROUND_LEFT, SURROUND_RIGHT,
BACK_LEFT, BACK_RIGHT,
TOP_FRONT_LEFT, TOP_FRONT_RIGHT,
TOP_BACK_LEFT, TOP_BACK_RIGHT,
CUSTOM
};

struct Speaker {
SpeakerType type;
Vec3 position;
bool hasElevation;
bool isLFE;
std::string name;

Speaker(SpeakerType t, Vec3 pos, bool elev = false, bool lfe = false, std::string n = "")
: type(t), position(pos), hasElevation(elev), isLFE(lfe), name(n) {}
};

class SpeakerArray {
private:
std::vector<Speaker> speakers;
int numSpeakers;

public:
SpeakerArray() : numSpeakers(0) {}

void setupDolbyAtmos714() {
speakers.clear();
speakers.push_back({SpeakerType::LEFT, Vec3(-0.707f, 0, 0.707f), false, false, "L"});
speakers.push_back({SpeakerType::RIGHT, Vec3(0.707f, 0, 0.707f), false, false, "R"});
speakers.push_back({SpeakerType::CENTER, Vec3(0, 0, 1), false, false, "C"});
speakers.push_back({SpeakerType::LFE, Vec3(0, -0.5f, 0.5f), false, true, "LFE"});
speakers.push_back({SpeakerType::SURROUND_LEFT, Vec3(-1, 0, -0.5f), false, false, "SL"});
speakers.push_back({SpeakerType::SURROUND_RIGHT, Vec3(1, 0, -0.5f), false, false, "SR"});
speakers.push_back({SpeakerType::BACK_LEFT, Vec3(-0.707f, 0, -0.707f), false, false, "BL"});
speakers.push_back({SpeakerType::BACK_RIGHT, Vec3(0.707f, 0, -0.707f), false, false, "BR"});
float elevation = 0.707f;
speakers.push_back({SpeakerType::TOP_FRONT_LEFT, Vec3(-0.5f, elevation, 0.5f), true, false, "TFL"});
speakers.push_back({SpeakerType::TOP_FRONT_RIGHT, Vec3(0.5f, elevation, 0.5f), true, false, "TFR"});
speakers.push_back({SpeakerType::TOP_BACK_LEFT, Vec3(-0.5f, elevation, -0.5f), true, false, "TBL"});
speakers.push_back({SpeakerType::TOP_BACK_RIGHT, Vec3(0.5f, elevation, -0.5f), true, false, "TBR"});
numSpeakers = speakers.size();
}

void setupCircular(int count) {
speakers.clear();
numSpeakers = std::min(count, Config::MAX_SPEAKERS);
for (int i = 0; i < numSpeakers; i++) {
float angle = (float)i / numSpeakers * 2.0f * 3.14159265f;
Vec3 pos(std::sin(angle), 0, std::cos(angle));
speakers.push_back({SpeakerType::CUSTOM, pos, false, false, "SP" + std::to_string(i)});
}
}

std::vector<float> calculateGains(float azimuth, float elevation) const {
std::vector<Vec3> positions;
for (const auto& spk : speakers) {
positions.push_back(spk.position);
}
return SpatialMatrix::calculateVBAP(azimuth, elevation, positions);
}

int getCount() const { return numSpeakers; }
const std::vector<Speaker>& getSpeakers() const { return speakers; }
};

// ============================================================================
// WASAPI OUTPUT
// ============================================================================

class WASAPIOutput {
private:
IMMDeviceEnumerator* enumerator = nullptr;
IMMDevice* device = nullptr;
IAudioClient* audioClient = nullptr;
IAudioRenderClient* renderClient = nullptr;
WAVEFORMATEXTENSIBLE waveFormat;
UINT32 bufferFrameCount;
int numChannels;
bool isExclusive = false;

public:
bool initialize(int channels) {
numChannels = channels;
HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);

hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&enumerator);
if (FAILED(hr)) return false;

hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
if (FAILED(hr)) return false;

hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL,
nullptr, (void**)&audioClient);
if (FAILED(hr)) return false;

WAVEFORMATEX* mixFormat = nullptr;
hr = audioClient->GetMixFormat(&mixFormat);
if (FAILED(hr)) return false;

ZeroMemory(&waveFormat, sizeof(WAVEFORMATEXTENSIBLE));
waveFormat.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
waveFormat.Format.nChannels = std::min(channels, (int)mixFormat->nChannels);
waveFormat.Format.nSamplesPerSec = Config::SAMPLE_RATE;
waveFormat.Format.wBitsPerSample = 32;
waveFormat.Format.nBlockAlign = waveFormat.Format.nChannels * 4;
waveFormat.Format.nAvgBytesPerSec = waveFormat.Format.nSamplesPerSec * waveFormat.Format.nBlockAlign;
waveFormat.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
waveFormat.Samples.wValidBitsPerSample = 32;
waveFormat.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;

if (mixFormat->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
waveFormat.dwChannelMask = ((WAVEFORMATEXTENSIBLE*)mixFormat)->dwChannelMask;
} else {
waveFormat.dwChannelMask = (1 << waveFormat.Format.nChannels) - 1;
}

hr = audioClient->Initialize(
AUDCLNT_SHAREMODE_EXCLUSIVE, 0,
10000000, 10000000,
(WAVEFORMATEX*)&waveFormat, nullptr);

if (SUCCEEDED(hr)) {
isExclusive = true;
} else {
hr = audioClient->Initialize(
AUDCLNT_SHAREMODE_SHARED, 0,
10000000, 0, mixFormat, nullptr);
if (FAILED(hr)) {
CoTaskMemFree(mixFormat);
return false;
}
}

CoTaskMemFree(mixFormat);

hr = audioClient->GetBufferSize(&bufferFrameCount);
if (FAILED(hr)) return false;

hr = audioClient->GetService(__uuidof(IAudioRenderClient), (void**)&renderClient);
if (FAILED(hr)) return false;

hr = audioClient->Start();
return SUCCEEDED(hr);
}

bool write(const std::vector<std::vector<float>>& channelData) {
if (!renderClient || channelData.empty()) return false;

UINT32 numFramesPadding;
HRESULT hr = audioClient->GetCurrentPadding(&numFramesPadding);
if (FAILED(hr)) return false;

UINT32 numFramesAvailable = bufferFrameCount - numFramesPadding;
if (numFramesAvailable == 0) return true;

BYTE* data;
hr = renderClient->GetBuffer(numFramesAvailable, &data);
if (FAILED(hr)) return false;

float* floatData = (float*)data;
int framesToWrite = std::min((int)numFramesAvailable, (int)channelData[0].size());
int channels = std::min((int)channelData.size(), numChannels);

for (int frame = 0; frame < framesToWrite; frame++) {
for (int ch = 0; ch < channels; ch++) {
*floatData++ = channelData[ch][frame];
}
for (int ch = channels; ch < numChannels; ch++) {
*floatData++ = 0.0f;
}
}

renderClient->ReleaseBuffer(framesToWrite, 0);
return true;
}

void shutdown() {
if (audioClient) audioClient->Stop();
if (renderClient) renderClient->Release();
if (audioClient) audioClient->Release();
if (device) device->Release();
if (enumerator) enumerator->Release();
CoUninitialize();
}

bool isExclusiveMode() const { return isExclusive; }
};

// ============================================================================
// WASAPI CAPTURE (LOOPBACK)
// ============================================================================

class WASAPICapture {
private:
IMMDeviceEnumerator* enumerator = nullptr;
IMMDevice* device = nullptr;
IAudioClient* audioClient = nullptr;
IAudioCaptureClient* captureClient = nullptr;
WAVEFORMATEX* waveFormat = nullptr;
int targetSampleRate;
float resampleRatio;

public:
WASAPICapture() : targetSampleRate(Config::SAMPLE_RATE), resampleRatio(1.0f) {}

bool initialize() {
HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);

hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&enumerator);
if (FAILED(hr)) return false;

hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
if (FAILED(hr)) return false;

hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL,
nullptr, (void**)&audioClient);
if (FAILED(hr)) return false;

hr = audioClient->GetMixFormat(&waveFormat);
if (FAILED(hr) || !waveFormat) return false;

resampleRatio = (float)targetSampleRate / (float)waveFormat->nSamplesPerSec;

hr = audioClient->Initialize(
AUDCLNT_SHAREMODE_SHARED,
AUDCLNT_STREAMFLAGS_LOOPBACK,
10000000, 0, waveFormat, nullptr);

if (FAILED(hr)) return false;

hr = audioClient->GetService(__uuidof(IAudioCaptureClient),
(void**)&captureClient);
if (FAILED(hr)) return false;

hr = audioClient->Start();
return SUCCEEDED(hr);
}

int captureChunk(float* outputBuffer, int maxSamples, std::vector<std::vector<float>>* multiChannel = nullptr) {
if (!captureClient || !waveFormat) return 0;

UINT32 packetLength = 0;
HRESULT hr = captureClient->GetNextPacketSize(&packetLength);
if (FAILED(hr) || packetLength == 0) return 0;

BYTE* data;
UINT32 numFrames;
DWORD flags;

hr = captureClient->GetBuffer(&data, &numFrames, &flags, nullptr, nullptr);
if (FAILED(hr) || !data) return 0;

bool isFloat = false;
if (waveFormat->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) {
isFloat = true;
} else if (waveFormat->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
WAVEFORMATEXTENSIBLE* wfext = (WAVEFORMATEXTENSIBLE*)waveFormat;
isFloat = (wfext->SubFormat.Data1 == 0x00000003);
}

int channels = waveFormat->nChannels;

// If multi-channel output is requested, populate it
if (multiChannel) {
multiChannel->resize(channels);
for (int ch = 0; ch < channels; ch++) {
(*multiChannel)[ch].resize(numFrames);
}
}

std::vector<float> tempBuffer(numFrames);

if (isFloat) {
float* floatData = (float*)data;
for (UINT32 i = 0; i < numFrames; i++) {
float sum = 0.0f;
for (int ch = 0; ch < channels; ch++) {
float sample = floatData[i * channels + ch];
sum += sample;
if (multiChannel) {
(*multiChannel)[ch][i] = sample;
}
}
tempBuffer[i] = sum / channels;
}
} else {
int16_t* intData = (int16_t*)data;
for (UINT32 i = 0; i < numFrames; i++) {
float sum = 0.0f;
for (int ch = 0; ch < channels; ch++) {
float sample = intData[i * channels + ch] / 32768.0f;
sum += sample;
if (multiChannel) {
(*multiChannel)[ch][i] = sample;
}
}
tempBuffer[i] = sum / channels;
}
}

int outputSamples = (int)(numFrames * resampleRatio);
outputSamples = std::min(outputSamples, maxSamples);

if (std::abs(resampleRatio - 1.0f) < 0.01f) {
memcpy(outputBuffer, tempBuffer.data(), outputSamples * sizeof(float));
} else {
for (int i = 0; i < outputSamples; i++) {
float srcPos = i / resampleRatio;
int srcIdx = (int)srcPos;
float frac = srcPos - srcIdx;
if (srcIdx + 1 < (int)numFrames) {
outputBuffer[i] = tempBuffer[srcIdx] * (1.0f - frac) +
tempBuffer[srcIdx + 1] * frac;
} else {
outputBuffer[i] = tempBuffer[srcIdx];
}
}
}

captureClient->ReleaseBuffer(numFrames);
return outputSamples;
}

void shutdown() {
if (audioClient) audioClient->Stop();
if (captureClient) captureClient->Release();
if (audioClient) audioClient->Release();
if (device) device->Release();
if (enumerator) enumerator->Release();
if (waveFormat) CoTaskMemFree(waveFormat);
CoUninitialize();
}
};

// ============================================================================
// WINDOWS SCREEN CAPTURE (GDI)
// ============================================================================

class ScreenCapture {
private:
HDC hdcScreen;
HDC hdcMemory;
HBITMAP hBitmap;
BITMAPINFO bmpInfo;
std::vector<uint8_t> pixelData;
int width, height;
std::mutex captureMutex;

public:
ScreenCapture() : hdcScreen(nullptr), hdcMemory(nullptr), hBitmap(nullptr),
width(Config::CV_WIDTH), height(Config::CV_HEIGHT) {}

bool initialize() {
hdcScreen = GetDC(NULL);
if (!hdcScreen) return false;

hdcMemory = CreateCompatibleDC(hdcScreen);
if (!hdcMemory) return false;

hBitmap = CreateCompatibleBitmap(hdcScreen, width, height);
if (!hBitmap) return false;

SelectObject(hdcMemory, hBitmap);

ZeroMemory(&bmpInfo, sizeof(BITMAPINFO));
bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
bmpInfo.bmiHeader.biWidth = width;
bmpInfo.bmiHeader.biHeight = -height; // Top-down
bmpInfo.bmiHeader.biPlanes = 1;
bmpInfo.bmiHeader.biBitCount = 32;
bmpInfo.bmiHeader.biCompression = BI_RGB;

pixelData.resize(width * height * 4);

return true;
}

bool captureFrame(uint8_t* outputBuffer) {
std::lock_guard<std::mutex> lock(captureMutex);

if (!hdcScreen || !hdcMemory) return false;

// Capture full screen and scale down
int screenWidth = GetSystemMetrics(SM_CXSCREEN);
int screenHeight = GetSystemMetrics(SM_CYSCREEN);

StretchBlt(hdcMemory, 0, 0, width, height,
hdcScreen, 0, 0, screenWidth, screenHeight,
SRCCOPY);

GetDIBits(hdcMemory, hBitmap, 0, height, pixelData.data(),
&bmpInfo, DIB_RGB_COLORS);

// Convert BGRA to RGB
for (int i = 0; i < width * height; i++) {
outputBuffer[i * 3 + 0] = pixelData[i * 4 + 2]; // R
outputBuffer[i * 3 + 1] = pixelData[i * 4 + 1]; // G
outputBuffer[i * 3 + 2] = pixelData[i * 4 + 0]; // B
}

return true;
}

void shutdown() {
if (hBitmap) DeleteObject(hBitmap);
if (hdcMemory) DeleteDC(hdcMemory);
if (hdcScreen) ReleaseDC(NULL, hdcScreen);
}

int getWidth() const { return width; }
int getHeight() const { return height; }
};

// ============================================================================
// CV-ENHANCED SPATIAL AUDIO ENGINE
// ============================================================================

class CVSpatialAudioEngine {
private:
WASAPICapture capture;
WASAPIOutput output;
SpeakerArray speakers;
MotionDetector motionDetector;
SurroundDetector surroundDetector;
AudioMotionCorrelator audioMotionCorrelator;
ScreenCapture screenCapture;

std::vector<DelayLine> delayLines;
std::vector<BiquadFilter> bandFilters;
std::vector<float> inputBuffer;
std::vector<std::vector<float>> outputBuffers;
std::vector<std::vector<float>> capturedChannels;

Vec3 listenerPosition;
std::atomic<bool> running{false};
std::atomic<bool> cvEnabled{true};
std::atomic<bool> cvActive{false};
std::thread audioThread;
std::thread cvThread;

struct Stats {
std::atomic<uint64_t> framesProcessed{0};
std::atomic<uint64_t> cvFramesProcessed{0};
std::atomic<float> peakLevel{0.0f};
std::atomic<int> activeObjects{0};
std::atomic<bool> surroundDetected{false};
} stats;

public:
CVSpatialAudioEngine(bool useDolbyAtmos = true, int circularSpeakers = 8, bool enableCV = true)
: listenerPosition(0.0f, 1.5f, 0.0f), cvEnabled(enableCV) {

if (useDolbyAtmos) {
speakers.setupDolbyAtmos714();
} else {
speakers.setupCircular(circularSpeakers);
}

int spkCount = speakers.getCount();
inputBuffer.resize(Config::CHUNK_SIZE);
delayLines.resize(spkCount);
bandFilters.resize(spkCount * Config::NUM_OCTAVE_BANDS);
outputBuffers.resize(spkCount);

for (int spk = 0; spk < spkCount; spk++) {
outputBuffers[spk].resize(Config::CHUNK_SIZE, 0.0f);
for (int band = 0; band < Config::NUM_OCTAVE_BANDS; band++) {
float freq = 62.5f * std::pow(2.0f, band);
bandFilters[spk * Config::NUM_OCTAVE_BANDS + band].setBandPass(
freq, 0.707f, Config::SAMPLE_RATE);
}
}
}

bool initialize() {
printf("\n╔═══════════════════════════════════════════════════════╗\n");
printf("║ CV-Enhanced Spatial Audio Engine ║\n");
printf("╚═══════════════════════════════════════════════════════╝\n\n");

if (!capture.initialize()) {
printf("[ERROR] Audio capture initialization failed\n");
return false;
}

if (!output.initialize(speakers.getCount())) {
printf("[ERROR] Audio output initialization failed\n");
return false;
}

if (cvEnabled) {
if (!screenCapture.initialize()) {
printf("[WARN] Screen capture failed, CV disabled\n");
cvEnabled = false;
} else {
printf("[CV] Screen capture initialized (%dx%d)\n",
screenCapture.getWidth(), screenCapture.getHeight());
}
}

printf("[ENGINE] Initialized with %d speakers\n", speakers.getCount());
printf("[CV] Computer vision: %s\n", cvEnabled.load() ? "ENABLED" : "DISABLED");
printf("[CV] Automatic surround detection: ACTIVE\n");

return true;
}

void start() {
running = true;
stats.framesProcessed.store(0);
stats.cvFramesProcessed.store(0);
stats.peakLevel.store(0.0f);
stats.activeObjects.store(0);
stats.surroundDetected.store(false);

printf("\n[ENGINE] Starting audio processing...\n");

// Audio processing thread
audioThread = std::thread([this]() {
HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
if (FAILED(hr)) return;

while (running) {
processAudioFrame();
std::this_thread::sleep_for(std::chrono::microseconds(100));
}

CoUninitialize();
});

// CV processing thread (if enabled)
if (cvEnabled) {
cvThread = std::thread([this]() {
std::vector<uint8_t> frameBuffer(
screenCapture.getWidth() * screenCapture.getHeight() * 3);

while (running) {
if (cvActive && screenCapture.captureFrame(frameBuffer.data())) {
motionDetector.processFrame(
frameBuffer.data(),
screenCapture.getWidth(),
screenCapture.getHeight(),
3
);
stats.cvFramesProcessed++;
stats.activeObjects = motionDetector.getTrackedObjects().size();
}
std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
}
});
}
}

void stop() {
running = false;
if (audioThread.joinable()) audioThread.join();
if (cvThread.joinable()) cvThread.join();
}

void shutdown() {
stop();
capture.shutdown();
output.shutdown();
screenCapture.shutdown();

printf("\n╔═══════════════════════════════════════════════════════╗\n");
printf("║ FINAL STATISTICS ║\n");
printf("╠═══════════════════════════════════════════════════════╣\n");
printf("║ Audio frames processed: %-28llu║\n",
(unsigned long long)stats.framesProcessed.load());
printf("║ CV frames processed: %-31llu║\n",
(unsigned long long)stats.cvFramesProcessed.load());
printf("║ Peak audio level: %-34.3f║\n", stats.peakLevel.load());
printf("║ Max tracked objects: %-31d║\n", stats.activeObjects.load());
printf("║ Surround detected: %-33s║\n",
stats.surroundDetected.load() ? "YES" : "NO");
printf("╚═══════════════════════════════════════════════════════╝\n");
}

void enableCV(bool enable) { cvEnabled = enable; }
const Stats& getStats() const { return stats; }

private:
void processAudioFrame() {
int samplesRead = capture.captureChunk(inputBuffer.data(),
Config::CHUNK_SIZE,
&capturedChannels);
if (samplesRead == 0) return;

stats.framesProcessed++;

// Analyze for surround sound detection
if (!capturedChannels.empty()) {
surroundDetector.analyzeChannels(capturedChannels);
stats.surroundDetected = surroundDetector.isTrueSurround();

// Auto-disable CV if true surround detected
if (surroundDetector.isTrueSurround() && cvActive) {
cvActive = false;
printf("\n[CV] True surround detected - CV auto-disabled\n");
} else if (surroundDetector.shouldEnableCV() && cvEnabled && !cvActive) {
cvActive = true;
printf("\n[CV] Fake/no surround detected - CV auto-enabled\n");
}
}

// Calculate audio energy
float audioEnergy = 0.0f;
float peak = 0.0f;
for (int i = 0; i < samplesRead; i++) {
audioEnergy += inputBuffer[i] * inputBuffer[i];
peak = std::max(peak, std::abs(inputBuffer[i]));
}
audioEnergy /= samplesRead;
stats.peakLevel = peak;

// Clear output buffers
for (auto& buf : outputBuffers) {
std::fill(buf.begin(), buf.end(), 0.0f);
}

// Write to delay lines
for (size_t spk = 0; spk < outputBuffers.size(); spk++) {
for (int i = 0; i < samplesRead; i++) {
delayLines[spk].write(inputBuffer[i]);
}
}

// Determine sound source position
Vec3 sourcePosition(0, 1.5f, 2.0f); // Default forward
float azimuth = 0.0f;
float elevation = 0.0f;

if (cvActive && audioEnergy > 1e-6f) {
// Use CV to determine sound source
auto trackedObjects = motionDetector.getTrackedObjects();

if (!trackedObjects.empty()) {
// Calculate motion energy
Vec2 motionDir = motionDetector.estimatePrimaryMotionDirection();
float motionEnergy = motionDir.length();

// Correlate with audio
float correlation = audioMotionCorrelator.correlate(audioEnergy, motionEnergy);

if (correlation > 0.3f && !trackedObjects.empty()) {
// Find most correlated object
int bestIdx = 0;
float bestCorr = 0.0f;

for (size_t i = 0; i < trackedObjects.size(); i++) {
if (trackedObjects[i].isActive && trackedObjects[i].framesTracked > 5) {
float objMotion = trackedObjects[i].velocity.length();
float objCorr = objMotion * correlation;
if (objCorr > bestCorr) {
bestCorr = objCorr;
bestIdx = i;
}
}
}

if (bestCorr > 0.1f) {
// Convert screen position to 3D audio position
float screenX = trackedObjects[bestIdx].box.x +
trackedObjects[bestIdx].box.width / 2.0f;
float screenY = trackedObjects[bestIdx].box.y +
trackedObjects[bestIdx].box.height / 2.0f;

// Normalize to -1 to 1
float normX = (screenX / screenCapture.getWidth()) * 2.0f - 1.0f;
float normY = (screenY / screenCapture.getHeight()) * 2.0f - 1.0f;

// Convert to azimuth and elevation
azimuth = normX * 90.0f; // ±90 degrees
elevation = -normY * 45.0f; // ±45 degrees (inverted Y)

// Update source position
float azRad = azimuth * 3.14159f / 180.0f;
float elRad = elevation * 3.14159f / 180.0f;
float distance = 2.0f;

sourcePosition = Vec3(
distance * std::sin(azRad) * std::cos(elRad),
distance * std::sin(elRad),
distance * std::cos(azRad) * std::cos(elRad)
);
}
}
}
}

// Calculate speaker gains using consolidated VBAP
auto speakerGains = speakers.calculateGains(azimuth, elevation);

// Apply spatial processing
float distance = (sourcePosition - listenerPosition).length();

for (size_t spk = 0; spk < outputBuffers.size(); spk++) {
float delayTime = distance / Config::SPEED_OF_SOUND * Config::SAMPLE_RATE;

for (int i = 0; i < samplesRead; i++) {
float delayed = delayLines[spk].read(delayTime);

// Multi-band processing
float sample = 0.0f;
for (int band = 0; band < Config::NUM_OCTAVE_BANDS; band++) {
float freq = 62.5f * std::pow(2.0f, band);
float bandSample = bandFilters[spk * Config::NUM_OCTAVE_BANDS + band].process(delayed);

// Apply consolidated transfer function
float transferCoeff = SpatialMatrix::calculateTransferCoefficient(
freq, azimuth, elevation, distance, 0.3f, 1);

sample += bandSample * transferCoeff;
}

// Apply speaker gain
sample *= speakerGains[spk];

// Soft clipping
sample = std::tanh(sample * 0.9f);

outputBuffers[spk][i] = sample;
}
}

// Write to output
output.write(outputBuffers);

// Status update
if (stats.framesProcessed % 100 == 0) {
printf("\r[MONITOR] Audio: %llu | CV: %llu | Objects: %d | Peak: %.3f | Surround: %s | CV: %s ",
(unsigned long long)stats.framesProcessed.load(),
(unsigned long long)stats.cvFramesProcessed.load(),
stats.activeObjects.load(),
stats.peakLevel.load(),
stats.surroundDetected.load() ? "YES" : "NO",
cvActive.load() ? "ACTIVE" : "INACTIVE");
fflush(stdout);
}
}
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
bool useDolbyAtmos = false;
int circularSpeakers = 8;
bool enableCV = true;
bool interactiveMode = true;

// Parse arguments
for (int i = 1; i < argc; i++) {
if (strcmp(argv[i], "--dolby") == 0) {
useDolbyAtmos = true;
interactiveMode = false;
} else if (strcmp(argv[i], "--speakers") == 0 && i + 1 < argc) {
circularSpeakers = atoi(argv[i + 1]);
interactiveMode = false;
i++;
} else if (strcmp(argv[i], "--no-cv") == 0) {
enableCV = false;
interactiveMode = false;
} else if (strcmp(argv[i], "--cv") == 0) {
enableCV = true;
interactiveMode = false;
}
}

// Interactive configuration
if (interactiveMode) {
printf("\n╔═══════════════════════════════════════════════════════╗\n");
printf("║ Arby Ultimate - CV-Enhanced Spatial Audio ║\n");
printf("╚═══════════════════════════════════════════════════════╝\n\n");

printf("Speaker configuration:\n");
printf(" 1. Dolby Atmos 7.1.4 (12 speakers with elevation)\n");
printf(" 2. Circular 360° (8-32 speakers)\n");
printf("Choose (1 or 2, default 1): ");

char input[100];
if (fgets(input, sizeof(input), stdin)) {
int choice = atoi(input);
if (choice == 2) {
printf("Number of speakers (8-32, default 8): ");
if (fgets(input, sizeof(input), stdin)) {
int parsed = atoi(input);
if (parsed >= 8 && parsed <= 32) {
circularSpeakers = parsed;
}
}
useDolbyAtmos = false;
} else {
useDolbyAtmos = true;
}
}

printf("\nEnable Computer Vision motion tracking?\n");
printf(" 1. Yes - Track moving objects on screen\n");
printf(" 2. No - Traditional audio processing only\n");
printf("Choose (1 or 2, default 1): ");
if (fgets(input, sizeof(input), stdin)) {
int choice = atoi(input);
enableCV = (choice != 2);
}

printf("\n═══════════════════════════════════════════════════════\n");
printf("Configuration Summary:\n");
printf("═══════════════════════════════════════════════════════\n");
printf(" • Mode: %s\n", useDolbyAtmos ? "Dolby Atmos 7.1.4" : "Circular 360°");
if (!useDolbyAtmos) {
printf(" • Speakers: %d\n", circularSpeakers);
}
printf(" • Computer Vision: %s\n", enableCV ? "ENABLED" : "DISABLED");
printf(" • Surround Detection: AUTO\n");
printf(" • Motion Tracking: %s\n", enableCV ? "ENABLED" : "DISABLED");
printf(" • Screen Capture: %s\n", enableCV ? "ENABLED" : "DISABLED");
printf(" • Multi-band filters: 7 octaves\n");
printf(" • Consolidated spatial algebra: ACTIVE\n");
printf("═══════════════════════════════════════════════════════\n\n");
}

CVSpatialAudioEngine engine(useDolbyAtmos, circularSpeakers, enableCV);

if (!engine.initialize()) {
printf("[ERROR] Engine initialization failed\n");
return -1;
}

engine.start();

printf("\n╔═══════════════════════════════════════════════════════╗\n");
printf("║ ENGINE RUNNING ║\n");
printf("╠═══════════════════════════════════════════════════════╣\n");
printf("║ Features Active: ║\n");
if (useDolbyAtmos) {
printf("║ ✓ Dolby Atmos 7.1.4 with elevation ║\n");
} else {
printf("║ ✓ Circular 360° speaker array ║\n");
}
printf("║ ✓ Consolidated linear algebra (VBAP + transfer) ║\n");
printf("║ ✓ Multi-band filtering (7 octaves) ║\n");
printf("║ ✓ Automatic surround detection ║\n");
if (enableCV) {
printf("║ ✓ Computer vision motion tracking ║\n");
printf("║ ✓ Audio-motion correlation ║\n");
printf("║ ✓ Automatic CV enable/disable based on surround ║\n");
}
printf("║ ║\n");
printf("║ Monitoring: ║\n");
printf("║ • Audio frames processed ║\n");
printf("║ • CV frames processed (if enabled) ║\n");
printf("║ • Tracked objects count ║\n");
printf("║ • Peak audio level ║\n");
printf("║ • Surround sound detection status ║\n");
printf("║ • CV active/inactive status ║\n");
printf("╚═══════════════════════════════════════════════════════╝\n");
printf("\nPress Enter to stop...\n\n");

getchar();

printf("\n\n╔═══════════════════════════════════════════════════════╗\n");
printf("║ SHUTTING DOWN ║\n");
printf("╚═══════════════════════════════════════════════════════╝\n");

engine.shutdown();

printf("\n═══════════════════════════════════════════════════════\n");
printf("Technology Highlights:\n");
printf("═══════════════════════════════════════════════════════\n");
printf("✓ Consolidated spatial processing using linear algebra\n");
printf("✓ VBAP (Vector Base Amplitude Panning) for speaker gains\n");
printf("✓ Unified transfer function: H(ω,θ,φ,d,r)\n");
printf("✓ Built-in computer vision (no OpenCV dependency)\n");
printf("✓ Motion detection with object tracking\n");
printf("✓ Audio-motion cross-correlation\n");
printf("✓ Automatic surround sound detection\n");
printf("✓ Detection of fake surround (stereo upmix, mono upscale)\n");
printf("✓ Auto-enable/disable CV based on true surround\n");
printf("✓ Screen capture via Windows GDI\n");
printf("✓ Multi-band frequency processing\n");
printf("✓ Distance-based attenuation with air absorption\n");
printf("✓ Soft clipping for distortion prevention\n");
printf("═══════════════════════════════════════════════════════\n\n");

printf("Thank you for using Arby Ultimate!\n\n");

return 0;
}
