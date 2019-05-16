
df = pd.read_csv('../data/letter-recognition.csv', header=None)
df['TARGET'] = df[0]
df = df.drop([0], axis=1)

def sample_df(df, num_letters=6):
    letters = [chr(x+65) for x in np.random.choice(26, num_letters, replace=False)]
    rows = np.any([df.TARGET == l for l in letters], axis=0)
    df = df[rows].apply(lambda c: c.astype('category'))
    return df

D5 = sample_df(df, 5)
D6 = sample_df(df, 6)
D7 = sample_df(df, 7)
D8 = sample_df(df, 8)

import pickle
pickle.dump([D5, D6, D7, D8], open('letter.pkl', 'wb'))
