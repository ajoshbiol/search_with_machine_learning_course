import fasttext

threshold = .75

if __name__ == '__main__':
    model = fasttext.load_model('/workspace/datasets/fasttext/title_model_m100_e25.bin')
    with open('/workspace/datasets/fasttext/top_words.txt') as f:
        for line in f:
            word = line.strip()
            nn = model.get_nearest_neighbors(word)
            output_line_arr = [word]
            for neighbor in nn:
                if neighbor[0] > threshold:
                    output_line_arr.append(neighbor[1])
            l = ','.join(output_line_arr)
            print(l)