#include <chrono>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

class MessageWatcher {
public:
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::milliseconds;

    // burstThreshold: how many occurrences within burstWindow count as "repeated"
    // burstWindow: sliding time window to count occurrences in
    // silenceTimeout: how long without the message before we declare it "gone quiet"
    MessageWatcher(std::string target,
                   size_t burstThreshold,
                   Duration burstWindow,
                   Duration silenceTimeout)
        : target_(std::move(target)),
          burstThreshold_(burstThreshold),
          burstWindow_(burstWindow),
          silenceTimeout_(silenceTimeout),
          lastSeen_(Clock::now()) {}

    // Callbacks — set these before use
    std::function<void(size_t count)> onBurstDetected;
    std::function<void(Duration sinceLast)> onSilenceDetected;

    // Call this every time you receive a message (any message).
    // It only reacts if msg == target_.
    void feed(const std::string& msg) {
        auto now = Clock::now();
        std::lock_guard<std::mutex> lock(mutex_);

        if (msg != target_) return;

        lastSeen_ = now;
        silenceFired_ = false; // reset — it's back, so silence alert can fire again later

        timestamps_.push_back(now);
        trimWindow(now);

        if (!burstFired_ && timestamps_.size() >= burstThreshold_) {
            burstFired_ = true;
            if (onBurstDetected) onBurstDetected(timestamps_.size());
        }

        // Once window empties back below threshold, allow burst to re-trigger later
        if (timestamps_.size() < burstThreshold_) {
            burstFired_ = false;
        }
    }

    // Call this periodically (e.g. in a timer/poll loop) regardless of
    // whether a message arrived, so silence can be detected even when
    // nothing at all is coming in.
    void poll() {
        auto now = Clock::now();
        std::lock_guard<std::mutex> lock(mutex_);

        trimWindow(now);

        auto sinceLast = std::chrono::duration_cast<Duration>(now - lastSeen_);
        if (!silenceFired_ && sinceLast >= silenceTimeout_) {
            silenceFired_ = true;
            if (onSilenceDetected) onSilenceDetected(sinceLast);
        }
    }

    // Optional: reset all state (e.g. after handling an alert)
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        timestamps_.clear();
        burstFired_ = false;
        silenceFired_ = false;
        lastSeen_ = Clock::now();
    }

private:
    void trimWindow(Clock::time_point now) {
        while (!timestamps_.empty() &&
               now - timestamps_.front() > burstWindow_) {
            timestamps_.pop_front();
        }
    }

    std::string target_;
    size_t burstThreshold_;
    Duration burstWindow_;
    Duration silenceTimeout_;

    std::deque<Clock::time_point> timestamps_;
    Clock::time_point lastSeen_;
    bool burstFired_ = false;
    bool silenceFired_ = false;

    std::mutex mutex_;
};

MessageWatcher watcher("HEARTBEAT",
                        /*burstThreshold=*/5,
                        /*burstWindow=*/std::chrono::milliseconds(2000),
                        /*silenceTimeout=*/std::chrono::milliseconds(10000));

watcher.onBurstDetected = [](size_t count) {
    std::cout << "Message repeated " << count << " times in the window!\n";
};

watcher.onSilenceDetected = [](std::chrono::milliseconds since) {
    std::cout << "No message seen for " << since.count() << "ms\n";
};

// In your ingest loop:
watcher.feed(incomingMessage);

// In a periodic timer (e.g. every 500ms), even if nothing arrives:
watcher.poll();
