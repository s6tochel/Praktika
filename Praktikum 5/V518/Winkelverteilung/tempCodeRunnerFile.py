for j in range(3):
    print(f"{j}:\t{fit_values[j]}\t± {fit_value_errors[j]}")
print(f"\tChi: {chi_squared}")