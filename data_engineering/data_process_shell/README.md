# Data Processing in Shell

## Download File

### CURL

* Filenames
  * -O to keep original filenames
  * -o to assign new filenames

* Redirected URL
  * -L to allow downloading from redirected URL
  * Download and rename the file in the same step
`curl -o Spotify201812.zip -L https://assets.datacamp.com/production/repositories/4180/datasets/eb1d6a36fa3039e4e00064797e1a1600d267b135/201812SpotifyData.zip`

* Download all 100 data files
`curl -O https://s3.amazonaws.com/assets.datacamp.com/production/repositories/4180/datasets/files/datafile[001-100].txt`

### WGET

* -c is for completing a previously incomplete download.
* -b is for downloading in the background.
* -i filename to iterate through the file to download multiple URLs
* --wait=n to pause specified time in seconds between downloads
* --limit-rate=xxxk to limit download speed in KB/s

## Data Processing with CSVKIT

### Installation

* `pip3 install csvkit`

### Convert to CSV file

* `in2csv filename > csv_filename.csv`
  * `in2csv filename.xlsx --sheet "sheet_name" > csv_filename.csv`

### View CSV file

* `csvlook csv_filename.csv`
* `csvstat csv_filename.csv`

### Print CSV column names

* `csvcut -n Spotify_MusicAttributes.csv`

### Print columns by position

* first column - `csvcut -c 1 filename.csv`
* first, third and fivth column - `csvcut -c 1,3,5 filename.csv`
* by name - `csvcut -c "track_id" filename.csv`
* by multile names - `csvcut -c "track_id","duration_ms","loudness" Spotify_MusicAttributes.csv`
  * no space between column names allowed

### Filter for row(s) where track_id = 118GQ70Sp6pMqn6w1oKuki

`csvgrep -c "track_id" -m 118GQ70Sp6pMqn6w1oKuki Spotify_MusicAttributes.csv`

## Stack the two files and save results as a new file

`csvstack SpotifyData_PopularityRank6.csv SpotifyData_PopularityRank7.csv > SpotifyPopularity.csv`

## Database Operation

### Query database

* `sql2csv -h`
* `sql2csv --db "sqlite:///SpotifyDatabase.db" --query "SELECT * FROM Spotify_Popularity"`

### Query CSV file

* `csvsql --query "SELECT * FROM Spotify_MusicAttributes ORDER BY duration_ms LIMIT 1" Spotify_MusicAttributes.csv`
* Store SQL query as shell variable - `sqlquery="SELECT * FROM Spotify_MusicAttributes ORDER BY duration_ms LIMIT 1"`
* Apply SQL query to Spotify_MusicAttributes.csv - `csvsql --query "$sqlquery" Spotify_MusicAttributes.csv`
* Upload Spotify_MusicAttributes.csv to database - `csvsql --db "sqlite:///SpotifyDatabase.db" --insert Spotify_MusicAttributes.csv`
