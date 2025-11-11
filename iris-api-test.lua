-- WRK Load Testing Script for Iris Prediction API
-- This script sends POST requests to the /predict endpoint with random Iris features

-- Initialize random seed
math.randomseed(os.time())

-- Sample Iris data for each species
local iris_samples = {
    -- Setosa samples
    { sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2 },
    { sepal_length = 4.9, sepal_width = 3.0, petal_length = 1.4, petal_width = 0.2 },
    { sepal_length = 5.4, sepal_width = 3.9, petal_length = 1.7, petal_width = 0.4 },
    
    -- Versicolor samples
    { sepal_length = 7.0, sepal_width = 3.2, petal_length = 4.7, petal_width = 1.4 },
    { sepal_length = 6.4, sepal_width = 3.2, petal_length = 4.5, petal_width = 1.5 },
    { sepal_length = 6.9, sepal_width = 3.1, petal_length = 4.9, petal_width = 1.5 },
    
    -- Virginica samples
    { sepal_length = 6.3, sepal_width = 3.3, petal_length = 6.0, petal_width = 2.5 },
    { sepal_length = 7.1, sepal_width = 3.0, petal_length = 5.9, petal_width = 2.1 },
    { sepal_length = 6.5, sepal_width = 3.0, petal_length = 5.8, petal_width = 2.2 }
}

-- Counter for tracking requests
request_count = 0
error_count = 0
success_count = 0

-- This function is called for each request
request = function()
    request_count = request_count + 1
    
    -- Select a random sample
    local sample = iris_samples[math.random(#iris_samples)]
    
    -- Create JSON payload
    local body = string.format(
        '{"sepal_length": %.1f, "sepal_width": %.1f, "petal_length": %.1f, "petal_width": %.1f}',
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width
    )
    
    -- Set headers and return request
    return wrk.format(
        "POST",
        "/predict",
        {
            ["Content-Type"] = "application/json",
            ["Accept"] = "application/json"
        },
        body
    )
end

-- This function is called for each response
response = function(status, headers, body)
    if status == 200 then
        success_count = success_count + 1
    else
        error_count = error_count + 1
        -- Uncomment to see error responses
        -- print("Error response:", status, body)
    end
end

-- This function is called when all requests are complete
done = function(summary, latency, requests)
    print("\n========================================")
    print("Load Test Summary")
    print("========================================")
    print(string.format("Total Requests:    %d", request_count))
    print(string.format("Successful:        %d (%.2f%%)", success_count, (success_count/request_count)*100))
    print(string.format("Failed:            %d (%.2f%%)", error_count, (error_count/request_count)*100))
    print(string.format("Duration:          %.2f seconds", summary.duration / 1000000))
    print(string.format("Requests/sec:      %.2f", summary.requests / (summary.duration / 1000000)))
    print(string.format("Transfer/sec:      %.2f KB", summary.bytes / (summary.duration / 1000000) / 1024))
    print("\n--- Latency Statistics ---")
    print(string.format("Min:               %.2f ms", latency.min / 1000))
    print(string.format("Max:               %.2f ms", latency.max / 1000))
    print(string.format("Mean:              %.2f ms", latency.mean / 1000))
    print(string.format("Stdev:             %.2f ms", latency.stdev / 1000))
    print(string.format("50th percentile:   %.2f ms", latency:percentile(50) / 1000))
    print(string.format("75th percentile:   %.2f ms", latency:percentile(75) / 1000))
    print(string.format("90th percentile:   %.2f ms", latency:percentile(90) / 1000))
    print(string.format("99th percentile:   %.2f ms", latency:percentile(99) / 1000))
    print("========================================\n")
end
