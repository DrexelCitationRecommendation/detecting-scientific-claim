# %%
file_path = 'dev.txt'
new_file_path = 'new_dev.txt'

# %%
new_file = open(new_file_path, 'w')
count = 0

with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('##'):
            count += 1
            line = line.replace('#', 'z')
            new_file.write(line)
        else:
            new_file.write(line)

# %%
new_file.close()

# %%
