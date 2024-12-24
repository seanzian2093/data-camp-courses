import random


class Playlist:
    def __init__(self, songs, shuffle=False):
        self.songs = songs
        self.index = 0

        if shuffle:
            random.shuffle(self.songs)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.songs):
            raise StopIteration
        song = self.songs[self.index]
        self.index += 1
        return song


if __name__ == "__main__":
    playlist = Playlist(["song1", "song2", "song3"], shuffle=True)
    for song in playlist:
        print(song)

    playlist = Playlist(["song1", "song2", "song3"], shuffle=True)
    while True:
        try:
            print(next(playlist))
        except StopIteration:
            print("Reached end of playlist")
            break
