import glob

input_file = glob.glob('train/HAZY/*.png')
output_file = glob.glob('train/GT/*.png')

print('input_files',input_file)
print('output_files',output_file)

with open('train_patch_hazy.txt', 'w+') as file:
	for input_path in input_file:
		file.write(input_path + '\n')

with open('train_patch_gt.txt', 'w+') as file1:
	for output_path in output_file:
		file1.write(output_path + '\n')
