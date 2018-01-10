from flickrapi import FlickrAPI  # https://pypi.python.org/pypi/flickrapi
import urllib
import os
import config
from random import randint
import time

def get_photos(keyword, size='original', max_nb_img=-1):
    
    flickr = FlickrAPI(config.API_KEY, config.API_SECRET)
        
    if size == 'thumbnail':
        size_url = 'url_t'
    elif size == 'square':
        size_url = 'url_q'
    elif size == 'medium':
        size_url = 'url_c'
    elif size == 'original':
        size_url = 'url_o'
    
    results_folder = config.IMG_FOLDER + keyword.replace(" ", "_") + "/"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    photos = flickr.walk(
                    text=keyword,
                    extras=size_url,
                    license='1,2,4,5',
                    per_page=50)
    
    urls = []
    count = 0
    print("Starting Flickr scraping for images with '" + keyword + "'")
    for photo in photos:
        t = randint(1, 3)
        time.sleep(t)
        count += 1
        if max_nb_img != -1:
            if count > max_nb_img:
                print('Reached maximum number of images to download')
                break
        try:
            url=photo.get(size_url)
            urls.append(url)
            
            urllib.request.urlretrieve(url,  results_folder + str(count) +".jpg")
            print('Downloading image #' + str(count) + ' from url ' + url)
        except Exception as e:
            print(e, 'Download failure')
                            
    print("Total images downloaded:", str(count - 1))