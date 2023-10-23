dataset = [
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "Yes"},
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "Yes"},
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "Yes"},
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "Yes"},
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "No"},
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "No"},
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "No"},
    {"Day": "Weekday", "Discount": "No", "Delivery": "Yes", "Purchase": "No"},
]

# Calculate the prior probabilities
total_samples = len(dataset)
purchase_count = sum(1 for data in dataset if data["Purchase"] == "Yes")
not_purchase_count = total_samples - purchase_count

prior_purchase = purchase_count / total_samples
prior_not_purchase = not_purchase_count / total_samples

# Define the input data for prediction
input_data = {"Day": "Weekday", "Discount": "No", "Delivery": "Yes"}

# Calculate conditional probabilities
conditional_purchase = 1
conditional_not_purchase = 1

for feature, value in input_data.items():
    purchase_given_feature = sum(1 for data in dataset if data[feature] == value and data["Purchase"] == "Yes")
    not_purchase_given_feature = sum(1 for data in dataset if data[feature] == value and data["Purchase"] == "No")
    
    conditional_purchase *= (purchase_given_feature / purchase_count)
    conditional_not_purchase *= (not_purchase_given_feature / not_purchase_count)

# Apply Bayes' theorem to make a prediction
posterior_purchase = (prior_purchase * conditional_purchase)
posterior_not_purchase = (prior_not_purchase * conditional_not_purchase)

# Normalize the probabilities
total_posterior = posterior_purchase + posterior_not_purchase
posterior_purchase /= total_posterior
posterior_not_purchase /= total_posterior

# Print the results
print("Probability of Purchase:", posterior_purchase)
print("Probability of Not Purchase:", posterior_not_purchase)