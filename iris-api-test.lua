    -- WRK Load Testing Script for Iris Prediction API
-- This script sends POST requests to the /predict endpoint with random Iris features

-- Initialize random seed
math.randomseed(os.time())

local iris_samples = {
    { sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2 },
    { sepal_length = 4.9, sepal_width = 3.0, petal_length = 1.4, petal_width = 0.2 },
    { sepal_length = 5.4, sepal_width = 3.9, petal_length = 1.7, petal_width = 0.4 },
    { sepal_length = 7.0, sepal_width = 3.2, petal_length = 4.7, petal_width = 1.4 },
    { sepal_length = 6.4, sepal_width = 3.2, petal_length = 4.5, petal_width = 1.5 },
    { sepal_length = 6.9, sepal_width = 3.1, petal_length = 4.9, petal_width = 1.5 },
    { sepal_length = 6.3, sepal_width = 3.3, petal_length = 6.0, petal_width = 2.5 },
    { sepal_length = 7.1, sepal_width = 3.0, petal_length = 5.9, petal_width = 2.1 },
    { sepal_length = 6.5, sepal_width = 3.0, petal_length = 5.8, petal_width = 2.2 }
}

request = function()
    local sample = iris_samples[math.random(#iris_samples)]
    
    local body = string.format(
        '{"sepal_length": %.1f, "sepal_width": %.1f, "petal_length": %.1f, "petal_width": %.1f}',
        sample.sepal_length, sample.sepal_width, sample.petal_length, sample.petal_width
    )
    
    return wrk.format("POST", "/predict", {
        ["Content-Type"] = "application/json",
        ["Accept"] = "application/json"
    }, body)
end

done = function(summary, latency, requests)
    local total_requests = summary.requests
    local error_count = summary.errors.connect + summary.errors.read + summary.errors.write + summary.errors.status + summary.errors.timeout
    local success_count = total_requests - error_count
    
    print("\n========================================")
    print("Load Test Summary")
    print("========================================")
    print(string.format("Total Requests:    %d", total_requests))
    print(string.format("Successful:        %d (%.2f%%)", success_count, (success_count/total_requests)*100))
    print(string.format("Failed:            %d (%.2f%%)", error_count, (error_count/total_requests)*100))
    print(string.format("Duration:          %.2f seconds", summary.duration / 1000000))
    print(string.format("Requests/sec:      %.2f", summary.requests / (summary.duration / 1000000)))
    print(string.format("Data transferred:  %.2f MB", summary.bytes / (1024 * 1024)))
    print("\n--- Latency Statistics ---")
    print(string.format("Min:               %.2f ms", latency.min / 1000))
    print(string.format("Max:               %.2f ms", latency.max / 1000))
    print(string.format("Mean:              %.2f ms", latency.mean / 1000))
    print(string.format("Stdev:             %.2f ms", latency.stdev / 1000))
    print(string.format("50th percentile:   %.2f ms", latency:percentile(50) / 1000))
    print(string.format("75th percentile:   %.2f ms", latency:percentile(75) / 1000))
    print(string.format("90th percentile:   %.2f ms", latency:percentile(90) / 1000))
    print(string.format("99th percentile:   %.2f ms", latency:percentile(99) / 1000))
    print("\n--- Error Breakdown ---")
    print(string.format("Connect errors:    %d", summary.errors.connect))
    print(string.format("Read errors:       %d", summary.errors.read))
    print(string.format("Write errors:      %d", summary.errors.write))
    print(string.format("Status errors:     %d", summary.errors.status))
    print(string.format("Timeout errors:    %d", summary.errors.timeout))
    print("========================================\n")
end
