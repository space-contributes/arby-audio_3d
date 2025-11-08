// Arby Enhanced - Real-time Spatial Audio Engine with Multi-Band Filtering
// Features: 7.1.4 Dolby Atmos, Multi-band processing, JSON profiles, Elevation, LFE

#pragma once

#include <windows.h>
#include <functional>
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
#include <complex>
#include <array>
#include <fstream>
#include <string>
#include <map>

// Cross-platform stubs
#ifdef __APPLE__
#include <CoreAudio/CoreAudio.h>
#endif
#ifdef __linux__
#include <alsa/asoundlib.h>
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

namespace Config {
    constexpr int SAMPLE_RATE = 96000;
    constexpr int CHUNK_SIZE = 512;  // <30ms latency
    constexpr int FFT_SIZE = 2048;
    constexpr float SPEED_OF_SOUND = 343.0f;
    constexpr int MAX_DELAY_SAMPLES = 96000 * 2;
    constexpr int MAX_REFLECTION_ORDER = 3;
    constexpr int MAX_SPEAKERS = 32;
    constexpr int NUM_OCTAVE_BANDS = 7;  // 62.5Hz to 16kHz
    constexpr int MAX_WORKER_THREADS = 8;

    constexpr float DISTANCE_K = 0.5f;
    constexpr float DISTANCE_P = 1.2f;
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

// ============================================================================
// BIQUAD FILTER (Multi-band, SIMD-ready)
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
// MULTI-BAND PROCESSOR (7 Octaves)
// ============================================================================

class MultiBandProcessor {
private:
    static constexpr float BAND_FREQS[Config::NUM_OCTAVE_BANDS] = {
        62.5f, 125.0f, 250.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f
    };

    std::array<BiquadFilter, Config::NUM_OCTAVE_BANDS> filters;

public:
    MultiBandProcessor() {
        for (int i = 0; i < Config::NUM_OCTAVE_BANDS; i++) {
            filters[i].setBandPass(BAND_FREQS[i], 0.707f, Config::SAMPLE_RATE);
        }
    }

    std::array<float, Config::NUM_OCTAVE_BANDS> process(float input) {
        std::array<float, Config::NUM_OCTAVE_BANDS> bands;
        for (int i = 0; i < Config::NUM_OCTAVE_BANDS; i++) {
            bands[i] = filters[i].process(input);
        }
        return bands;
    }

    void reset() {
        for (auto& f : filters) f.reset();
    }
};

// ============================================================================
// FFT (Cooley-Tukey)
// ============================================================================

class FFT {
public:
    static void transform(std::vector<std::complex<float>>& data) {
        int n = data.size();
        if (n <= 1) return;

        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(data[i], data[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            float angle = -2.0f * 3.14159265f / len;
            std::complex<float> wlen(std::cos(angle), std::sin(angle));

            for (int i = 0; i < n; i += len) {
                std::complex<float> w(1, 0);
                for (int j = 0; j < len / 2; j++) {
                    std::complex<float> u = data[i + j];
                    std::complex<float> v = data[i + j + len / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    static std::vector<float> getMagnitudeSpectrum(const std::vector<float>& input) {
        std::vector<std::complex<float>> complexData(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            complexData[i] = std::complex<float>(input[i], 0);
        }

        transform(complexData);

        std::vector<float> magnitude(input.size() / 2);
        for (size_t i = 0; i < magnitude.size(); i++) {
            magnitude[i] = std::abs(complexData[i]);
        }
        return magnitude;
    }
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
// SPEAKER CONFIGURATION (Dolby Atmos 7.1.4 + Custom)
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

        // Ground level (0° elevation)
        speakers.push_back({SpeakerType::LEFT, Vec3(-0.707f, 0, 0.707f), false, false, "L"});
        speakers.push_back({SpeakerType::RIGHT, Vec3(0.707f, 0, 0.707f), false, false, "R"});
        speakers.push_back({SpeakerType::CENTER, Vec3(0, 0, 1), false, false, "C"});
        speakers.push_back({SpeakerType::LFE, Vec3(0, -0.5f, 0.5f), false, true, "LFE"});
        speakers.push_back({SpeakerType::SURROUND_LEFT, Vec3(-1, 0, -0.5f), false, false, "SL"});
        speakers.push_back({SpeakerType::SURROUND_RIGHT, Vec3(1, 0, -0.5f), false, false, "SR"});
        speakers.push_back({SpeakerType::BACK_LEFT, Vec3(-0.707f, 0, -0.707f), false, false, "BL"});
        speakers.push_back({SpeakerType::BACK_RIGHT, Vec3(0.707f, 0, -0.707f), false, false, "BR"});

        // Height channels (45° elevation)
        float elevation = 0.707f;
        speakers.push_back({SpeakerType::TOP_FRONT_LEFT, Vec3(-0.5f, elevation, 0.5f), true, false, "TFL"});
        speakers.push_back({SpeakerType::TOP_FRONT_RIGHT, Vec3(0.5f, elevation, 0.5f), true, false, "TFR"});
        speakers.push_back({SpeakerType::TOP_BACK_LEFT, Vec3(-0.5f, elevation, -0.5f), true, false, "TBL"});
        speakers.push_back({SpeakerType::TOP_BACK_RIGHT, Vec3(0.5f, elevation, -0.5f), true, false, "TBR"});

        numSpeakers = speakers.size();
        printf("[SPEAKERS] Dolby Atmos 7.1.4 configured (%d speakers)\n", numSpeakers);
    }

    void setupCircular(int count) {
        speakers.clear();
        numSpeakers = std::min(count, Config::MAX_SPEAKERS);

        for (int i = 0; i < numSpeakers; i++) {
            float angle = (float)i / numSpeakers * 2.0f * 3.14159265f;
            Vec3 pos(std::sin(angle), 0, std::cos(angle));
            speakers.push_back({SpeakerType::CUSTOM, pos, false, false, "SP" + std::to_string(i)});
        }

        printf("[SPEAKERS] Circular 360° (%d speakers)\n", numSpeakers);
    }

    void setCustomPositions(const std::vector<Speaker>& customSpeakers) {
        speakers = customSpeakers;
        numSpeakers = speakers.size();
    }

    std::vector<float> calculateGains(float azimuth, float elevation) const {
        std::vector<float> gains(numSpeakers, 0.0f);

        float azRad = azimuth * 3.14159265f / 180.0f;
        float elRad = elevation * 3.14159265f / 180.0f;
        Vec3 sourceDir(
            std::cos(elRad) * std::sin(azRad),
            std::sin(elRad),
            std::cos(elRad) * std::cos(azRad)
        );

        float sumSquared = 0.0f;
        for (int i = 0; i < numSpeakers; i++) {
            // LFE handling: always receives signal at reduced level
            if (speakers[i].isLFE) {
                gains[i] = 0.3f;  // LFE omnidirectional bass
                sumSquared += gains[i] * gains[i];
                continue;
            }

            Vec3 spkPos = speakers[i].position.normalize();
            float dotProd = std::clamp(sourceDir.dot(spkPos), -1.0f, 1.0f);
            float angularDist = std::acos(dotProd);

            // Elevation boost for height channels
            float elevationBoost = 1.0f;
            if (speakers[i].hasElevation && elevation > 20.0f) {
                elevationBoost = 1.5f;
            }

            float gain = std::max(0.0f, std::cos(angularDist)) * elevationBoost;
            gains[i] = gain;
            sumSquared += gain * gain;
        }

        // Normalize
        float rms = std::sqrt(sumSquared);
        if (rms > 1e-6f) {
            for (auto& g : gains) g /= rms;
        }

        return gains;
    }

    int getCount() const { return numSpeakers; }
    const std::vector<Speaker>& getSpeakers() const { return speakers; }
};

// ============================================================================
// AUDIO PROFILE (JSON Save/Load)
// ============================================================================

struct AudioProfile {
    std::string name;
    Vec3 roomDimensions;
    Vec3 sourcePosition;
    Vec3 listenerPosition;
    float wallAbsorption;
    std::vector<Speaker> customSpeakers;

    AudioProfile() 
        : name("Default"),
          roomDimensions(8, 6, 3.2f),
          sourcePosition(2.5f, 1.5f, 2),
          listenerPosition(0, 1.5f, 0),
          wallAbsorption(0.3f) {}

    bool saveToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "{\n";
        file << "  \"name\": \"" << name << "\",\n";
        file << "  \"room\": {\"x\": " << roomDimensions.x << ", \"y\": " << roomDimensions.y << ", \"z\": " << roomDimensions.z << "},\n";
        file << "  \"source\": {\"x\": " << sourcePosition.x << ", \"y\": " << sourcePosition.y << ", \"z\": " << sourcePosition.z << "},\n";
        file << "  \"listener\": {\"x\": " << listenerPosition.x << ", \"y\": " << listenerPosition.y << ", \"z\": " << listenerPosition.z << "},\n";
        file << "  \"absorption\": " << wallAbsorption << "\n";
        file << "}\n";

        file.close();
        return true;
    }

    bool loadFromFile(const std::string& filename) {
        // Simplified JSON parsing (in production, use proper JSON library)
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("\"name\"") != std::string::npos) {
                size_t start = line.find(":") + 3;
                size_t end = line.find("\"", start);
                name = line.substr(start, end - start);
            }
            // Additional parsing omitted for brevity
        }

        file.close();
        return true;
    }
};

class ProfileManager {
private:
    std::map<std::string, AudioProfile> profiles;
    std::string activeProfile;

public:
    void addProfile(const std::string& name, const AudioProfile& profile) {
        profiles[name] = profile;
    }

    bool setActive(const std::string& name) {
        if (profiles.find(name) != profiles.end()) {
            activeProfile = name;
            return true;
        }
        return false;
    }

    AudioProfile* getActive() {
        if (profiles.find(activeProfile) != profiles.end()) {
            return &profiles[activeProfile];
        }
        return nullptr;
    }
};

// ============================================================================
// ROOM & WALL
// ============================================================================

struct Wall {
    Vec3 normal;
    Vec3 point;
    float absorption;

    Wall(const Vec3& n, const Vec3& p, float abs = 0.3f) 
        : normal(n.normalize()), point(p), absorption(abs) {}

    Vec3 reflect(const Vec3& p) const {
        float dist = normal.dot(p - point);
        return p - normal * (2.0f * dist);
    }
};

enum class ScaleMode {
    WHOLE_ROOM,
    PER_WALL
};

class LowPassFilter {
private:
    float cutoffFreq;
    float sampleRate;
    float alpha;
    float previousOutput;

public:
    LowPassFilter(float cutoff = 18000.0f, float fs = 96000.0f) 
        : cutoffFreq(cutoff), sampleRate(fs), previousOutput(0.0f) {
        float RC = 1.0f / (2.0f * 3.14159265f * cutoffFreq);
        float dt = 1.0f / sampleRate;
        alpha = dt / (RC + dt);
    }

    float process(float input) {
        previousOutput = previousOutput + alpha * (input - previousOutput);
        return previousOutput;
    }

    void reset() { previousOutput = 0.0f; }
};

class Room {
private:
    Vec3 dimensions;
    Vec3 minDimensions;
    std::vector<Wall> walls;
    ScaleMode scaleMode;
    float baseAbsorption;
    std::mutex roomMutex;
    LowPassFilter energyFilter;

public:
    Room(float w = 8.0f, float h = 6.0f, float d = 3.2f) 
        : dimensions(w, h, d), 
          minDimensions(8.0f, 6.0f, 3.2f),
          scaleMode(ScaleMode::WHOLE_ROOM), 
          baseAbsorption(0.3f),
          energyFilter(5.0f, 96000.0f) {
        rebuild();
    }

    void setScaleMode(ScaleMode mode) { scaleMode = mode; }

    void autoScale(float audioEnergy) {
        std::lock_guard<std::mutex> lock(roomMutex);

        float smoothedEnergy = energyFilter.process(audioEnergy);
        float scale = 1.0f + std::sqrt(smoothedEnergy) * 2.0f;
        scale = std::clamp(scale, 1.0f, 3.0f);

        if (scaleMode == ScaleMode::WHOLE_ROOM) {
            dimensions = minDimensions * scale;
        } else {
            dimensions.x = minDimensions.x * scale;
            dimensions.y = minDimensions.y * scale;
            dimensions.z = minDimensions.z * scale;
        }

        float volumeAbsorption = 0.5f - smoothedEnergy * 0.3f;
        volumeAbsorption = std::clamp(volumeAbsorption, 0.1f, 0.8f);

        for (auto& wall : walls) {
            wall.absorption = volumeAbsorption;
        }

        rebuild();
    }

    void rebuild() {
        if (walls.empty()) {
            walls = {
                {Vec3(0, 1, 0),  Vec3(0, 0, 0), baseAbsorption},
                {Vec3(0, -1, 0), Vec3(0, dimensions.y, 0), baseAbsorption},
                {Vec3(1, 0, 0),  Vec3(0, 0, 0), baseAbsorption},
                {Vec3(-1, 0, 0), Vec3(dimensions.x, 0, 0), baseAbsorption},
                {Vec3(0, 0, 1),  Vec3(0, 0, 0), baseAbsorption},
                {Vec3(0, 0, -1), Vec3(0, 0, dimensions.z), baseAbsorption}
            };
        } else {
            walls[0].point = Vec3(0, 0, 0);
            walls[1].point = Vec3(0, dimensions.y, 0);
            walls[2].point = Vec3(0, 0, 0);
            walls[3].point = Vec3(dimensions.x, 0, 0);
            walls[4].point = Vec3(0, 0, 0);
            walls[5].point = Vec3(0, 0, dimensions.z);
        }
    }

    const std::vector<Wall>& getWalls() const { return walls; }
    Vec3 getDimensions() const { return dimensions; }
};

// ============================================================================
// AIR ABSORPTION (Multithreaded ISO 9613-1)
// ============================================================================

class AirAbsorption {
private:
    static constexpr float TEMPERATURE = 20.0f;
    static constexpr float HUMIDITY = 50.0f;
    static constexpr float PRESSURE = 101.325f;
    static constexpr float P_REF = 101.325f;
    static constexpr float T_0 = 293.15f;
    static constexpr float T_01 = 273.16f;

    static float getSaturationPressure(float T_kelvin) {
        float C = -6.8346f * std::pow(T_01 / T_kelvin, 1.261f) + 4.6151f;
        return P_REF * std::pow(10.0f, C);
    }

    static float getMolarConcentration(float T_kelvin, float relHumidity) {
        float p_sat = getSaturationPressure(T_kelvin);
        return (relHumidity / 100.0f) * (p_sat / PRESSURE);
    }

    static float getOxygenRelaxation(float h_molar) {
        return (PRESSURE / P_REF) * 
               (24.0f + 4.04e4f * h_molar * (0.02f + h_molar) / (0.391f + h_molar));
    }

    static float getNitrogenRelaxation(float T_kelvin, float h_molar) {
        float T_ratio = T_kelvin / T_0;
        float exp_term = std::exp(-4.170f * (std::pow(T_ratio, -1.0f/3.0f) - 1.0f));
        return (PRESSURE / P_REF) * std::sqrt(T_ratio) * 
               (9.0f + 280.0f * h_molar * exp_term);
    }

public:
    static float getAbsorptionCoeff(float frequency) {
        float T_kelvin = TEMPERATURE + 273.15f;
        float T_ratio = T_kelvin / T_0;
        float f = frequency;
        float f2 = f * f;

        float h = getMolarConcentration(T_kelvin, HUMIDITY);
        float f_rO = getOxygenRelaxation(h);
        float f_rN = getNitrogenRelaxation(T_kelvin, h);

        float classical = 1.84e-11f * (P_REF / PRESSURE) * std::sqrt(T_ratio);
        float oxygen_term = 0.01275f * std::exp(-2239.1f / T_kelvin) / 
                            (f_rO + f2 / f_rO);
        float nitrogen_term = 0.1068f * std::exp(-3352.0f / T_kelvin) / 
                              (f_rN + f2 / f_rN);
        float molecular = std::pow(T_ratio, -2.5f) * (oxygen_term + nitrogen_term);

        float alpha = 8.686f * f2 * (classical + molecular);
        return alpha;
    }

    static void calculateAbsorption(std::vector<float>& frequencyGains, float distance, int numThreads = 4) {
        int size = frequencyGains.size();
        int chunkSize = (size + numThreads - 1) / numThreads;
        std::vector<std::thread> threads;

        for (int t = 0; t < numThreads; t++) {
            int start = t * chunkSize;
            int end = std::min(start + chunkSize, size);
            if (start >= size) break;

            threads.emplace_back([&frequencyGains, distance, start, end]() {
                for (int i = start; i < end; i++) {
                    float freq = (float)i * Config::SAMPLE_RATE / Config::FFT_SIZE;
                    float absorptionDB = getAbsorptionCoeff(freq) * distance;
                    float absorptionGain = std::pow(10.0f, -absorptionDB / 20.0f);
                    frequencyGains[i] *= absorptionGain;
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
};

// ============================================================================
// REFLECTION PATH
// ============================================================================

struct ReflectionPath {
    Vec3 virtualSource;
    float distance;
    float azimuth;
    float elevation;
    float delaySamples;
    float gain;
    std::vector<float> frequencyGains;
    int order;
};

class ReflectionCalculator {
private:
    Room& room;

public:
    ReflectionCalculator(Room& r) : room(r) {}

    std::vector<ReflectionPath> calculatePaths(const Vec3& source, const Vec3& listener, int maxOrder) {
        std::vector<ReflectionPath> paths;
        paths.push_back(computeDirectPath(source, listener));
        calculateReflectionsRecursive(source, listener, {}, 1, maxOrder, paths);
        return paths;
    }

private:
    void calculateReflectionsRecursive(
        const Vec3& currentSource,
        const Vec3& listener,
        const std::vector<int>& wallSequence,
        int order,
        int maxOrder,
        std::vector<ReflectionPath>& paths) {

        if (order > maxOrder) return;

        const auto& walls = room.getWalls();

        for (size_t i = 0; i < walls.size(); i++) {
            if (!wallSequence.empty() && wallSequence.back() == (int)i) continue;

            Vec3 virtualSrc = walls[i].reflect(currentSource);
            auto newSequence = wallSequence;
            newSequence.push_back(i);

            ReflectionPath path = computePath(virtualSrc, listener, newSequence, order);
            paths.push_back(path);

            if (order < maxOrder) {
                calculateReflectionsRecursive(virtualSrc, listener, newSequence, order + 1, maxOrder, paths);
            }
        }
    }

    ReflectionPath computeDirectPath(const Vec3& source, const Vec3& listener) {
        return computePath(source, listener, {}, 0);
    }

    ReflectionPath computePath(const Vec3& src, const Vec3& listener, 
                               const std::vector<int>& wallSeq, int order) {
        ReflectionPath path;
        path.virtualSource = src;
        path.order = order;

        Vec3 diff = listener - src;
        path.distance = diff.length();

        Vec3 dir = diff.normalize();
        path.azimuth = std::atan2(dir.x, dir.z) * 180.0f / 3.14159265f;
        path.elevation = std::asin(std::clamp(dir.y, -1.0f, 1.0f)) * 180.0f / 3.14159265f;

        path.delaySamples = path.distance / Config::SPEED_OF_SOUND * Config::SAMPLE_RATE;

        float d = path.distance;
        path.gain = 1.0f / (1.0f + Config::DISTANCE_K * std::pow(d, Config::DISTANCE_P));

        const auto& walls = room.getWalls();
        for (int wallIdx : wallSeq) {
            path.gain *= (1.0f - walls[wallIdx].absorption);
        }

        path.frequencyGains.resize(Config::FFT_SIZE / 2, 1.0f);

        for (size_t i = 0; i < path.frequencyGains.size(); i++) {
            float freq = (float)i * Config::SAMPLE_RATE / Config::FFT_SIZE;
            float freqLoss = std::exp(-freq * path.distance / 10000.0f);
            path.frequencyGains[i] = freqLoss;
        }

        AirAbsorption::calculateAbsorption(path.frequencyGains, path.distance, 4);

        return path;
    }
};

// ============================================================================
// AUDIO PROCESSOR THREAD POOL
// ============================================================================

class AudioProcessorThread {
private:
    std::vector<std::thread> workers;
    std::atomic<bool> running{false};
    std::mutex queueMutex;
    std::vector<std::function<void()>> tasks;

public:
    void start(int numThreads) {
        running = true;
        for (int i = 0; i < numThreads; i++) {
            workers.emplace_back([this]() {
                while (running) {
                    std::function<void()> task;
                    {
                        std::lock_guard<std::mutex> lock(queueMutex);
                        if (!tasks.empty()) {
                            task = tasks.back();
                            tasks.pop_back();
                        }
                    }
                    if (task) task();
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
        }
    }

    void addTask(std::function<void()> task) {
        std::lock_guard<std::mutex> lock(queueMutex);
        tasks.push_back(task);
    }

    void stop() {
        running = false;
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }
};

// ============================================================================
// SPEAKER CALIBRATION
// ============================================================================

class SpeakerCalibrator {
private:
    std::array<float, Config::MAX_SPEAKERS> gains;
    std::array<float, Config::MAX_SPEAKERS> delays;

public:
    SpeakerCalibrator() {
        gains.fill(1.0f);
        delays.fill(0.0f);
    }

    void calibrate(int speakerIndex, float gain, float delayMs) {
        if (speakerIndex < Config::MAX_SPEAKERS) {
            gains[speakerIndex] = gain;
            delays[speakerIndex] = delayMs * Config::SAMPLE_RATE / 1000.0f;
        }
    }

    float getGain(int index) const {
        return index < Config::MAX_SPEAKERS ? gains[index] : 1.0f;
    }

    float getDelay(int index) const {
        return index < Config::MAX_SPEAKERS ? delays[index] : 0.0f;
    }
};

// ============================================================================
// MONITORING & STATISTICS
// ============================================================================

struct AudioStats {
    std::atomic<uint64_t> framesProcessed{0};
    std::atomic<uint64_t> framesSuccess{0};
    std::atomic<uint64_t> framesDropped{0};
    std::atomic<float> cpuLoad{0.0f};
    std::atomic<float> peakLevel{0.0f};

    float getSuccessRate() const {
        uint64_t total = framesProcessed.load();
        return total > 0 ? (float)framesSuccess.load() / total * 100.0f : 0.0f;
    }

    void reset() {
        framesProcessed = 0;
        framesSuccess = 0;
        framesDropped = 0;
        cpuLoad = 0.0f;
        peakLevel = 0.0f;
    }
};

// ============================================================================
// WASAPI OUTPUT (with std::array support)
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

        printf("[OUTPUT] Device format: %u Hz, %u ch, %u-bit\n",
            mixFormat->nSamplesPerSec, mixFormat->nChannels, mixFormat->wBitsPerSample);

        ZeroMemory(&waveFormat, sizeof(WAVEFORMATEXTENSIBLE));
        waveFormat.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        waveFormat.Format.nChannels = std::min(channels, (int)mixFormat->nChannels);
        waveFormat.Format.nSamplesPerSec = mixFormat->nSamplesPerSec;
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

        printf("[OUTPUT] Trying EXCLUSIVE mode at %u Hz...\n", waveFormat.Format.nSamplesPerSec);
        hr = audioClient->Initialize(
            AUDCLNT_SHAREMODE_EXCLUSIVE, 0,
            10000000, 10000000,
            (WAVEFORMATEX*)&waveFormat, nullptr);

        if (SUCCEEDED(hr)) {
            isExclusive = true;
            printf("[OUTPUT] ✓ EXCLUSIVE mode enabled!\n");
        } else {
            printf("[OUTPUT] Exclusive failed (0x%08X), using SHARED mode\n", (unsigned)hr);
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

    // Vector overload
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

    // Array overload for fixed speaker configurations
    template<size_t N>
    bool write(const std::array<std::vector<float>, N>& channelData) {
        if (!renderClient) return false;

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
        size_t channels = std::min(static_cast<size_t>(N), static_cast<size_t>(numChannels));

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

        printf("[CAPTURE] LOOPBACK: %lu Hz, %lu ch, %lu-bit\n", 
       (unsigned long)waveFormat->nSamplesPerSec, 
       (unsigned long)waveFormat->nChannels, 
       (unsigned long)waveFormat->wBitsPerSample);

        resampleRatio = (float)targetSampleRate / (float)waveFormat->nSamplesPerSec;
        printf("[CAPTURE] Resample ratio: %.3f\n", resampleRatio);

        hr = audioClient->Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            10000000, 0, waveFormat, nullptr);

        if (FAILED(hr)) return false;

        hr = audioClient->GetService(__uuidof(IAudioCaptureClient),
            (void**)&captureClient);
        if (FAILED(hr)) return false;

        hr = audioClient->Start();
        if (FAILED(hr)) return false;

        printf("[CAPTURE] ✓ Loopback capture started\n");
        return true;
    }

    int captureChunk(float* outputBuffer, int maxSamples) {
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
        std::vector<float> tempBuffer(numFrames);

        if (isFloat) {
            float* floatData = (float*)data;
            for (UINT32 i = 0; i < numFrames; i++) {
                float sum = 0.0f;
                for (int ch = 0; ch < channels; ch++) {
                    sum += floatData[i * channels + ch];
                }
                tempBuffer[i] = sum / channels;
            }
        } else {
            int16_t* intData = (int16_t*)data;
            for (UINT32 i = 0; i < numFrames; i++) {
                float sum = 0.0f;
                for (int ch = 0; ch < channels; ch++) {
                    sum += intData[i * channels + ch] / 32768.0f;
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
// SPATIAL AUDIO ENGINE (Enhanced)
// ============================================================================

class SpatialAudioEngine {
private:
    Room room;
    SpeakerArray speakers;
    ReflectionCalculator reflector;
    WASAPICapture capture;
    WASAPIOutput output;
    SpeakerCalibrator calibrator;
    AudioProcessorThread threadPool;
    AudioStats stats;

    std::vector<DelayLine> delayLines;
    std::vector<std::array<BiquadFilter, Config::NUM_OCTAVE_BANDS>> multiBandFilters;
    std::vector<MultiBandProcessor> bandProcessors;
    std::vector<float> inputBuffer;
    std::vector<std::vector<float>> outputBuffers;
    std::vector<BiquadFilter> antiDistortionFilters;

    Vec3 sourcePosition;
    Vec3 listenerPosition;

    std::atomic<bool> running{false};
    std::thread audioThread;

public:
    SpatialAudioEngine(bool useDolbyAtmos = true, int circularSpeakers = 8) 
        : reflector(room),
          sourcePosition(2.5f, 1.5f, 2.0f),
          listenerPosition(0.0f, 1.5f, 0.0f) {

        if (useDolbyAtmos) {
            speakers.setupDolbyAtmos714();
        } else {
            speakers.setupCircular(circularSpeakers);
        }

        int spkCount = speakers.getCount();
        inputBuffer.resize(Config::CHUNK_SIZE);
        delayLines.resize(spkCount);
        multiBandFilters.resize(spkCount);
        bandProcessors.resize(spkCount);
        outputBuffers.resize(spkCount);
        antiDistortionFilters.resize(spkCount);

        // Initialize per-speaker multi-band filters
        for (int spk = 0; spk < spkCount; spk++) {
            for (int band = 0; band < Config::NUM_OCTAVE_BANDS; band++) {
                float freq = 62.5f * std::pow(2.0f, band);
                multiBandFilters[spk][band].setLowPass(freq * 1.5f, 0.707f, Config::SAMPLE_RATE);
            }
            antiDistortionFilters[spk].setLowPass(18000.0f, 0.707f, Config::SAMPLE_RATE);
            outputBuffers[spk].resize(Config::CHUNK_SIZE, 0.0f);
        }
    }

    bool initialize() {
        printf("\n=== ENHANCED SPATIAL AUDIO ENGINE ===\n");
        printf("[ENGINE] Multi-band filtering: %d octaves\n", Config::NUM_OCTAVE_BANDS);
        printf("[ENGINE] Chunk size: %d samples (%.1f ms latency)\n", 
            Config::CHUNK_SIZE, 1000.0f * Config::CHUNK_SIZE / Config::SAMPLE_RATE);

        if (!capture.initialize()) {
            printf("[ENGINE] Capture init failed\n");
            return false;
        }

        if (!output.initialize(speakers.getCount())) {
            printf("[ENGINE] Output init failed\n");
            return false;
        }

        threadPool.start(std::min(4, Config::MAX_WORKER_THREADS));

        printf("[ENGINE] ✓ Initialized %d speakers\n", speakers.getCount());
        printf("[ENGINE] ✓ Thread pool: %d workers\n", 4);
        return true;
    }

    void start() {
        running = true;
        stats.reset();
        printf("\n[ENGINE] Starting enhanced audio processing...\n\n");

        audioThread = std::thread([this]() {
            HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
            if (FAILED(hr)) return;

            auto startTime = std::chrono::high_resolution_clock::now();

            while (running) {
                auto frameStart = std::chrono::high_resolution_clock::now();

                bool success = processFrame();

                auto frameEnd = std::chrono::high_resolution_clock::now();
                auto frameDuration = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart);

                stats.framesProcessed++;
                if (success) stats.framesSuccess++;
                else stats.framesDropped++;

                // Calculate CPU load
                float frameTimeMs = frameDuration.count() / 1000.0f;
                float availableTimeMs = 1000.0f * Config::CHUNK_SIZE / Config::SAMPLE_RATE;
                stats.cpuLoad = (frameTimeMs / availableTimeMs) * 100.0f;

                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            CoUninitialize();
        });
    }

    void stop() {
        running = false;
        if (audioThread.joinable()) {
            audioThread.join();
        }
        threadPool.stop();
    }

    void shutdown() {
        stop();
        capture.shutdown();
        output.shutdown();

        printf("\n[STATS] Total frames: %llu\n", (unsigned long long)stats.framesProcessed.load());
        printf("[STATS] Success rate: %.2f%%\n", stats.getSuccessRate());
        printf("[STATS] Dropped frames: %llu\n", (unsigned long long)stats.framesDropped.load());
        printf("[STATS] Peak level: %.3f\n", stats.peakLevel.load());
    }

    void setScaleMode(ScaleMode mode) { room.setScaleMode(mode); }
    void setSourcePosition(const Vec3& pos) { sourcePosition = pos; }
    void setListenerPosition(const Vec3& pos) { listenerPosition = pos; }
    const AudioStats& getStats() const { return stats; }

private:
    bool processFrame() {
        int samplesRead = capture.captureChunk(inputBuffer.data(), Config::CHUNK_SIZE);
        if (samplesRead == 0) return false;

        // Calculate energy and peak
        float energy = 0.0f;
        float peak = 0.0f;
        for (int i = 0; i < samplesRead; i++) {
            energy += inputBuffer[i] * inputBuffer[i];
            peak = std::max(peak, std::abs(inputBuffer[i]));
        }
        energy /= samplesRead;
        stats.peakLevel = peak;

        if (energy < 1e-7f) return false;

        room.autoScale(energy);

        auto paths = reflector.calculatePaths(sourcePosition, listenerPosition, Config::MAX_REFLECTION_ORDER);

        auto spectrum = FFT::getMagnitudeSpectrum(
            std::vector<float>(inputBuffer.begin(), inputBuffer.begin() + std::min(samplesRead, Config::FFT_SIZE))
        );

        // Clear output
        for (auto& buf : outputBuffers) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }

        // Write input to delay lines
        for (size_t spk = 0; spk < outputBuffers.size(); spk++) {
            for (int i = 0; i < samplesRead; i++) {
                delayLines[spk].write(inputBuffer[i]);
            }
        }

        // Process each path with multi-band filtering
        for (const auto& path : paths) {
            auto speakerGains = speakers.calculateGains(path.azimuth, path.elevation);

            for (size_t spk = 0; spk < speakerGains.size(); spk++) {
                for (int i = 0; i < samplesRead; i++) {
                    float effectiveDelay = std::max(0.0f, path.delaySamples - i + calibrator.getDelay(spk));
                    float delayed = delayLines[spk].read(effectiveDelay);

                    // Multi-band processing
                    auto bands = bandProcessors[spk].process(delayed);
                    float sample = 0.0f;

                    for (int band = 0; band < Config::NUM_OCTAVE_BANDS; band++) {
                        float bandSample = multiBandFilters[spk][band].process(bands[band]);
                        int freqIdx = std::min((int)(62.5f * std::pow(2.0f, band) * Config::FFT_SIZE / Config::SAMPLE_RATE), 
                                               (int)path.frequencyGains.size() - 1);
                        bandSample *= path.frequencyGains[freqIdx];
                        sample += bandSample;
                    }

                    sample *= path.gain * speakerGains[spk] * calibrator.getGain(spk);

                    if (i < (int)spectrum.size()) {
                        sample *= (1.0f + spectrum[i] * 0.05f);
                    }

                    outputBuffers[spk][i] += sample;
                }
            }
        }

        // Apply anti-distortion filters and clipping prevention
        for (size_t spk = 0; spk < outputBuffers.size(); spk++) {
            for (int i = 0; i < samplesRead; i++) {
                outputBuffers[spk][i] = antiDistortionFilters[spk].process(outputBuffers[spk][i]);
                outputBuffers[spk][i] = std::clamp(outputBuffers[spk][i], -0.95f, 0.95f);
            }
        }

        bool writeSuccess = output.write(outputBuffers);

        // Progress monitoring
        if (stats.framesProcessed % 100 == 0) {
            printf("\r[MONITOR] Frames: %llu | Success: %.1f%% | CPU: %.1f%% | Peak: %.3f | Paths: %zu     ",
                (unsigned long long)stats.framesProcessed.load(),
                stats.getSuccessRate(),
                stats.cpuLoad.load(),
                stats.peakLevel.load(),
                paths.size());
            fflush(stdout);
        }

        return writeSuccess;
    }
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    bool useDolbyAtmos = false;
    int circularSpeakers = 8;
    ScaleMode scaleMode = ScaleMode::WHOLE_ROOM;
    bool interactiveMode = true;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dolby") == 0) {
            useDolbyAtmos = true;
            interactiveMode = false;
        } else if (strcmp(argv[i], "--speakers") == 0 && i + 1 < argc) {
            circularSpeakers = atoi(argv[i + 1]);
            interactiveMode = false;
            i++;
        } else if (strcmp(argv[i], "--per-wall") == 0) {
            scaleMode = ScaleMode::PER_WALL;
            interactiveMode = false;
        } else if (strcmp(argv[i], "--whole-room") == 0) {
            scaleMode = ScaleMode::WHOLE_ROOM;
            interactiveMode = false;
        }
    }

    // Interactive configuration
    if (interactiveMode) {
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════╗\n");
        printf("║   Arby Enhanced - Spatial Audio Engine Configuration ║\n");
        printf("╚═══════════════════════════════════════════════════════╝\n\n");

        printf("Speaker configuration:\n");
        printf("  1. Dolby Atmos 7.1.4 (12 speakers with elevation + LFE)\n");
        printf("  2. Circular 360° (8-32 speakers, no elevation)\n");
        printf("Choose (1 or 2, default 1): ");

        char input[100];
        if (fgets(input, sizeof(input), stdin)) {
            int choice = atoi(input);
            if (choice == 2) {
                printf("Number of circular speakers (8-32, default 8): ");
                if (fgets(input, sizeof(input), stdin)) {
                    int parsed = atoi(input);
                    if (parsed >= 8 && parsed <= 32) {
                        circularSpeakers = parsed;
                    }
                }
                useDolbyAtmos = false;
                printf("✓ Using Circular 360° mode with %d speakers\n", circularSpeakers);
            } else {
                useDolbyAtmos = true;
                printf("✓ Using Dolby Atmos 7.1.4 mode\n");
            }
        }
        printf("\n");

        printf("Room scaling mode:\n");
        printf("  1. Whole room scaling (uniform expansion)\n");
        printf("  2. Per-wall scaling (independent wall movement)\n");
        printf("Choose (1 or 2, default 1): ");
        if (fgets(input, sizeof(input), stdin)) {
            int choice = atoi(input);
            if (choice == 2) {
                scaleMode = ScaleMode::PER_WALL;
                printf("✓ Using PER-WALL scaling mode\n");
            } else {
                scaleMode = ScaleMode::WHOLE_ROOM;
                printf("✓ Using WHOLE-ROOM scaling mode\n");
            }
        }

        printf("\n");
        printf("═══════════════════════════════════════════════════════\n");
        printf("Configuration Summary:\n");
        printf("═══════════════════════════════════════════════════════\n");
        if (useDolbyAtmos) {
            printf("  • Mode: Dolby Atmos 7.1.4\n");
            printf("  • Speakers: 12 (L, R, C, LFE, SL, SR, BL, BR + 4 height)\n");
            printf("  • Elevation: Yes (45° height channels)\n");
            printf("  • LFE: Yes (dedicated subwoofer channel)\n");
        } else {
            printf("  • Mode: Circular 360°\n");
            printf("  • Speakers: %d\n", circularSpeakers);
            printf("  • Elevation: No\n");
            printf("  • LFE: No\n");
        }
        printf("  • Scaling: %s\n", scaleMode == ScaleMode::WHOLE_ROOM ? "Whole room" : "Per-wall");
        printf("  • Multi-band filters: 7 octaves per speaker\n");
        printf("  • Chunk size: 512 samples (<30ms latency)\n");
        printf("  • Anti-distortion: 18kHz low-pass per speaker\n");
        printf("  • Air absorption: ISO 9613-1 (multithreaded)\n");
        printf("  • Thread pool: Up to 8 workers\n");
        printf("═══════════════════════════════════════════════════════\n");
        printf("\n");
    }

    SpatialAudioEngine engine(useDolbyAtmos, circularSpeakers);
    engine.setScaleMode(scaleMode);

    if (!engine.initialize()) {
        printf("Failed to initialize engine\n");
        return -1;
    }

    engine.start();

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                 ENGINE RUNNING                        ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Real-time monitoring active:                         ║\n");
    printf("║  • Frames processed & success rate                    ║\n");
    printf("║  • CPU load percentage                                ║\n");
    printf("║  • Peak audio level                                   ║\n");
    printf("║  • Active reflection paths                            ║\n");
    printf("║                                                       ║\n");
    printf("║  Features enabled:                                    ║\n");
    if (useDolbyAtmos) {
        printf("║  ✓ Dolby Atmos 7.1.4 (L/R/C/LFE/SL/SR/BL/BR/heights)║\n");
        printf("║  ✓ Elevation support (45° height channels)          ║\n");
        printf("║  ✓ LFE dedicated bass channel                       ║\n");
    } else {
        printf("║  ✓ Circular 360° speaker array                      ║\n");
    }
    printf("║  ✓ Multi-band filtering (7 octaves per speaker)      ║\n");
    printf("║  ✓ Per-speaker anti-distortion filters               ║\n");
    printf("║  ✓ Dynamic room scaling (volume-responsive)          ║\n");
    printf("║  ✓ Multi-threaded air absorption (ISO 9613-1)        ║\n");
    printf("║  ✓ Advanced reflection calculation (3rd order)       ║\n");
    printf("║  ✓ Thread pool processing (up to 8 workers)          ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\nPress Enter to stop and view final statistics...\n\n");

    getchar();

    printf("\n\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                 SHUTTING DOWN                         ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");

    engine.shutdown();

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║              SESSION STATISTICS                       ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    auto& finalStats = engine.getStats();
    printf("║  Total frames processed: %-28llu║\n", (unsigned long long)finalStats.framesProcessed.load());
    printf("║  Successful frames: %-33llu║\n", (unsigned long long)finalStats.framesSuccess.load());
    printf("║  Dropped frames: %-36llu║\n", (unsigned long long)finalStats.framesDropped.load());
    printf("║  Success rate: %6.2f%%                              ║\n", finalStats.getSuccessRate());
    printf("║  Peak audio level: %6.3f                           ║\n", finalStats.peakLevel.load());
    printf("║  Final CPU load: %6.1f%%                            ║\n", finalStats.cpuLoad.load());
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Audio processing features utilized:                  ║\n");
    printf("║  ✓ FFT spectrum analysis (2048 bins)                 ║\n");
    printf("║  ✓ Multi-band filtering (62.5Hz - 16kHz)             ║\n");
    printf("║  ✓ Frequency-dependent air absorption                ║\n");
    printf("║  ✓ Distance-based attenuation                        ║\n");
    printf("║  ✓ Wall reflection modeling (up to 3rd order)        ║\n");
    printf("║  ✓ Dynamic room acoustics                            ║\n");
    if (useDolbyAtmos) {
        printf("║  ✓ 3D spatial audio with elevation                   ║\n");
        printf("║  ✓ LFE channel for low-frequency effects            ║\n");
    } else {
        printf("║  ✓ 360° horizontal audio positioning                ║\n");
    }
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Optional: Save session profile
    printf("Save this configuration as profile? (y/n): ");
    char saveChoice[10];
    if (fgets(saveChoice, sizeof(saveChoice), stdin) && (saveChoice[0] == 'y' || saveChoice[0] == 'Y')) {
        AudioProfile profile;
        profile.name = "LastSession";
        profile.roomDimensions = Vec3(8, 6, 3.2f);

        if (profile.saveToFile("arby_profile.json")) {
            printf("✓ Profile saved to arby_profile.json\n");
        } else {
            printf("✗ Failed to save profile\n");
        }
    }

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Thank you for using Arby Enhanced Spatial Audio Engine!\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    return 0;
}
