from utils import generate_data, embed, plot

if __name__ == '__main__':
    data, colors = generate_data('swissroll')
    embedding = embed(data, method='lle')
    plot(data, embedding.T, colors)
