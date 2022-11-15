INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file yelp.inter recording the ratings of the users over the businesses.
Each record/line in the file has the following fields: review_id, user_id, business_id, stars, useful, funny, cool, date

review_id: the id of the review, and its type is token.
user_id: the id of the user, and its type is token.
business_id: the id of the business, and its type is token.
stars: star rating, and its type is float.
useful: number of useful votes received, and its type is float.
funny: number of funny votes received, and its type is float.
cool: number of cool votes received, and its type is float.
date: the UNIX timestamp of the creating time of the review, and its type is float.

BUSINESS DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file yelp.item recording the business information.
Each record/line in the file has the following fields: business_id, item_name, address, city, state, postal_code, latitude, longitude, item_stars, item_review_count, is_open, categories
 
business_id: the id of the business, and its type is token.
item_name: the business's name, and its type is token_seq.
address: the full address of the business, and its type is token_seq.
city: the city, and its type is token_seq.
state: 2 character state code, and its type is token.
postal_code: the postal code, and its type is token.
latitude: the latitude of the business, and its type is float.
longitude: the longitude of the business, and its type is float.
item_stars: star rating, and its type is float.
item_review_count: number of reviews, and its type is float.
is_open: 0 or 1 for closed or open, and its type is float.
categories: an array of strings of business categories, and its type is token_seq.

USER DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file yelp.user recording the user information.
Each record/line in the file has the following fields: user_id, user_name, user_review_count, yelping_since, user_useful, user_funny, user_cool, elite, fans, average_stars, compliment_hot, compliment_more, compliment_profile, compliment_cute, compliment_list, compliment_note, compliment_plain, compliment_cool, compliment_funny, compliment_writer, compliment_photos

user_id: the id of the user, and its type is token.
user_name: the user's first name, and its type is token.
user_review_count: the number of reviews they've written, and its type is float.
yelping_since: when the user joined Yelp, it has been converted to UNIX timestamp, and its type is float.
user_useful: number of useful votes sent by the user, and its type is float.
user_funny: number of funny votes sent by the user, and its type is float.
user_cool: number of cool votes sent by the user, and its type is float.
elite: array of integers, the years the user was elite, and its type is token.
fans: number of fans the user has, and its type is float.
average_stars: average rating of all reviews, and its type is float.
compliment_hot: number of hot compliments received by the user, and its type is float.
compliment_more: number of more compliments received by the user, and its type is float.
compliment_profile: number of profile compliments received by the user, and its type is float.
compliment_cute: number of cute compliments received by the user, and its type is float.
compliment_list: number of list compliments received by the user, and its type is float.
compliment_note: number of note compliments received by the user, and its type is float.
compliment_plain: number of plain compliments received by the user, and its type is float.
compliment_cool: number of cool compliments received by the user, and its type is float.
compliment_funny: number of funny compliments received by the user, and its type is float.
compliment_writer: number of writer compliments received by the user, and its type is float.
compliment_photos: number of photo compliments received by the user, and its type is float.
