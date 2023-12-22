import numpy as np
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt

# Parameters
n = 64  # number of bits
k = 4   # Derived from student numbers s1081911 and s1072878
noise_rate = 0.1
num_challenges = 20000
num_instances = 10
num_repetitions = 20

def simulate_puf_instance(seed):
    # Initialize the PUF instance
    instance = XORArbiterPUF(n=n, k=k, noisiness=noise_rate, seed=seed)
    # Generate random challenges
    challenges = random_inputs(n=n, N=num_challenges, seed=seed)
    # Collect responses for each measurement
    responses = np.array([instance.eval(challenges) for _ in range(num_repetitions)])
    return responses, challenges

# Simulate PUF instances and save responses
challenges = None
all_responses = []
for i in range(num_instances):
    seed = i  # Seed could be more specific
    responses, challenges = simulate_puf_instance(seed)
    all_responses.append(responses)
    # Save the responses to a .npy file
    np.save(f'puf_responses_instance_{i}.npy', responses)

# Challenges are the same for all instances and only need to be saved once
np.save('puf_challenges.npy', challenges)

# Redefine the compute_golden_response function to handle different numbers of measurements
all_responses = np.stack(all_responses, axis=0)
def compute_golden_response(responses, num_measurements):
    # Calculate majority-voted response for each PUF instance based on num_measurements
    golden_responses = []
    for response in responses:
        # Compute the majority vote across specified num_measurements
        votes = np.sum(response[:num_measurements, :], axis=0)
        majority_vote = votes > (num_measurements // 2)
        # Handle the ties by assigning a random decision
        ties = votes == (num_measurements // 2)
        majority_vote[ties] = np.random.choice([False, True], size=np.sum(ties))
        golden_response = np.where(majority_vote, 1, 0)
        golden_responses.append(golden_response)
    return np.array(golden_responses)

golden_response_q2 = compute_golden_response(all_responses, 15)

# Generate majority-voted responses for 5, 15, and 25 measurements and compute the HDs
measurement_counts = [5, 15, 25]
for count in measurement_counts:
    # Generate the temporal majority-voted response
    majority_voted_response = compute_golden_response(all_responses, count)
    
    # Compute the HD of the golden response from Q2 against the majority-voted response
    hd_from_majority = [hamming(majority_voted_response[i], golden_response_q2[i]) * n for i in range(num_instances)]
    
    # Check for identical responses
    if not np.any(hd_from_majority):
        print(f"Debug: All HD values are zero for {count} measurements. This may indicate an error.")
    
    # Print the HD results
    print(f"HD from majority-voted response with {count} measurements:", hd_from_majority)
    
    # Plot the histogram of the HD
    plt.hist(hd_from_majority, bins=20, alpha=0.75, color='green')
    plt.title(f'Histogram of HD from {count} Measurements')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Function to compute uniformity metric
def compute_uniformity(golden_responses):
    # Bias is the deviation from 50% ones (or zeros)
    biases = np.abs(0.5 - np.mean(golden_responses, axis=1))
    return biases

# Function to compute uniqueness metric
def compute_uniqueness(golden_responses, n):
    num_pairs = len(golden_responses)
    uniqueness = []
    for i in range(num_pairs):
        for j in range(i + 1, num_pairs):
            hd = hamming(golden_responses[i], golden_responses[j]) * n
            uniqueness.append(hd)
    return uniqueness

# Function to compute reliability metric
def compute_reliability(all_responses, golden_responses, n):
    reliability = []
    for i, responses in enumerate(all_responses):
        for response in responses:
            hd = hamming(golden_responses[i], response) * n
            reliability.append(hd)
    return reliability

# Calculate the golden responses for 20 repetitions (for Q2)
golden_responses = compute_golden_response(np.array(all_responses), 20)

# Calculate uniformity
uniformity_metrics = compute_uniformity(golden_responses)
print("Uniformity metrics:", uniformity_metrics)

# Plot the uniformity metrics
plt.hist(uniformity_metrics, bins=20, alpha=0.75, color='green')
plt.title('Histogram of Uniformity Metrics')
plt.xlabel('Bias')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Calculate uniqueness
uniqueness_metrics = compute_uniqueness(golden_responses, n)
print("Uniqueness metrics:", uniqueness_metrics)

# Calculate reliability
reliability_metrics = compute_reliability(all_responses, golden_responses, n)
print("Reliability metrics:", reliability_metrics)

# Save the metrics
np.save('uniformity_metrics.npy', uniformity_metrics)
np.save('uniqueness_metrics.npy', uniqueness_metrics)
np.save('reliability_metrics.npy', reliability_metrics)

# Plot the histogram of uniqueness metrics
plt.hist(uniqueness_metrics, bins=20, alpha=0.75, color='blue')
plt.title('Histogram of Uniqueness Metrics')
plt.xlabel('Hamming Distance')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

