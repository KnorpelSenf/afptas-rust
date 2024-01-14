import random

pairs = ["processing_time,resource_amount"]
for _ in range(100):
    first_number = random.uniform(0, 10)
    second_number = random.uniform(0, 5)
    pair = f"{first_number},{second_number}"
    pairs.append(pair)

pairs_string = "\n".join(pairs)
print(pairs_string)