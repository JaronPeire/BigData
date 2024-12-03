# Predicting Drivers Customer Engagement

## Introduction

Social media platforms serve as effective channels for building connections with customers. One specific strategy involves establishing brand fan pages on various social networking sites [1, 3]. These pages enable companies to share brand-related content such as videos, messages, quizzes, and information [1]. By creating brand posts on these fan pages, companies encourage customers to become fans and to engage with the content by expressing their approval through likes and comments [4]. This interaction, known as **brand post engagement**, plays a crucial role in fostering customer relationships [1]. In this study, we aim to identify potential drivers of brand post engagement by analyzing Instagram posts and activities of two airline companies: Lufthansa USA and Singapore Air.


## Objective

In this study, we aim to identify potential drivers of brand post engagement by analyzing and comparing Instagram posts and activities of both airline companies. By leveraging Big Data and machine learning techniques, we intend to empower companies to make data-driven decisions in managing their customer engagement.

## Methodology
We followed the CRISP-DM (Cross Industry Standard Process for Data Mining) framework [2] to streamline the machine learning process. 

# Installation and Setup

Below you may find the details of the Spark, Java and Python version our project was developped with and the packages necessary to run our Pyspark scripts. Note that every notebook includes a part where all necessary packages are imported.

## Codes and Resources Used
- **Spark Version:** Spark v3.5.0
- **Java version:** 21.0.1
- **Python Version:** 3.9

## Python Packages Used


- **General Packages**:`json`, `lzma`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `seaborn`

- **Sentiment Analysis**: `emoji`, `nltk`, `re`, `string.punctuation`, `textblob.TextBlob`, `vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer`

- **Image Analysis**`cv2`, `imutils`, `mtcnn`, `pytesseract`, `tensorflow.keras.applications.VGG16`, `tensorflow.keras.applications.vgg16.preprocess_input`, `tensorflow.keras.preprocessing.image`


# Data


## Source Data
Below, you may find the source files `lufthansa_usa`, `singaporeair` and their description. These were obtained from our course's 'Big Data' repository, and were in turn obtained through the Instagram API.

### `lufthansa_usa`


| Column          | Description                                        |
|-----------------|----------------------------------------------------|
| `%3Atagged`           | The tagged posts of Lufthansa USA (contains the same categories as below)      |
| `XXXX-XX-XX_xx-xx-xx_UTC`       | The unique identifier for every brand post   |
| `unique_id.jpg `       | The picture(s) of every brand post   |
| `unique_id.txt `       | The caption of every brand post   |
| `unique_id.mp4 `       | The video of a brand post (if one is posted)  |
| `unique_id.json`             | The metadata of every brand post         |
| `unique_id_comments.json`           | The metadata of the comments of every brand post product.  |

### `singaporeair`

| Column          | Description                                        |
|-----------------|----------------------------------------------------|
| `%3Atagged`           | The tagged posts of Singapore Air (contains the same categories as below)      |
| `XXXX-XX-XX_xx-xx-xx_UTC`       | The unique identifier for every brand post   |
| `unique_id.jpg `       | The picture(s) of every brand post   |
| `unique_id.txt `       | The caption of every brand post   |
| `unique_id.mp4 `       | The video of a brand post (if one is posted)  |
| `unique_id.json`             | The metadata of every brand post         |
| `unique_id_comments.json`           | The metadata of the comments of every brand post product.  |


## Data Preprocessing
Data cleaning involved enforcing temporal cut-offs starting from  `09-2016` so we could reasonably compare the two brands (they had about the same followers starting from this point) and handling missing values judiciously. String manipulation, such as extracting dates from post_id, ensured consistency in data representation.

## Feature Engineering
Features were derived from both Datasets, providing meaningful inputs for various models. In Table 1, you find an overview of all features used and their description.

| Feature Description       | Description                                                                        |
|---------------------------|------------------------------------------------------------------------------------|
| **Caption**               |                                                                                    |
| Contains City             | Indicates whether the caption contains the name of a city.                         |
| Nr. @                     | Counts the number of '@' mentions in the caption.                                  |
| Nr. ‘#’                   | Counts the number of '#' symbols in the caption.                                   |
| Nr. ‘?’                   | Counts the number of '?' symbols in the caption.                                   |
| Nr. ‘!’                   | Counts the number of '!' symbols in the caption.                                   |
| Contains Link             | Indicates whether the caption contains a hyperlink.                               |
| Sentiment                | Measures the sentiment of the caption using a sentiment analysis tool.            |
| Nr. Words                | Counts the number of words in the caption.                                         |
| Subjectivity            | Measures the subjectivity of the caption using a natural language processing tool.|
| Nr. Emojis               | Counts the number of emojis in the caption.                                       |
| **Post**                  |                                                                                    |
| Part of Day               | Identifies the part of the day (morning, afternoon, evening, night) in the post.  |
| Season                    | Determines the season the post was made in.                                      |
| Time Top Post             | Indicates how long a post was the first post on the brand's page.                         |
| Is Weekday               | Identifies whether the post was made on a weekday.                                |
| Location                  | Specifies the location where the post was made.                                  |
| Aspect Ratio              | Identifies if a post was `square`, `landscape`, or `portrait`.                    |
| Carousel                  | Indicates if multiple pictures were posted, and if so, how many.                   |
| Nr. Tags                  | Counts if a tag is present in the post.                                           |
| Vividness                | `High`: video. `Medium`: event/cell to act. `Low`: picture                       |
| **Image**                 |                                                                                    |
| Contains Airplane         | Indicates whether the image contains an airplane.                                 |
| Contains Logo             | Indicates whether the image contains the brand's logo.                            |
| Nr. Faces                 | Counts the number of faces detected in the image.                                 |
| Colourfulness            | Measures the colourfulness (RGB values) of the image content.                     |
| Contains Text             | Indicates whether the image contains textual information.                         |



**Table 1:** Feature Engineering Overview



# Code structure
Below you find the skeleton of our repository. 

- In the `1_general_and_strategic_brand_comparison` > `brand_comparison.ipynb`, you find the general comparison between both companies. 
- In `2_feature_engineering`, you find the code to go from our input data to our engineered features. An exploratory data analysis is included based on these features. 
- In the `3_modelling`, you find all the scripts regarding our modelling. 
- In the `data` > `lufthansa_usa` and `singaporeair`, you find the files before cleaning for both companies respectively. 
- In the `additional_data`, you find `.csv` files used by other notebooks. These are not important to regard for the user.
- In the `processed_data`, you find the basetables containing all features for the respective companies, on which the models were performed. These are not important to regard for the user.



```bash
├── 1_general_and_strategic_brand_comparison
│   └── brand_comparison.ipynb
├── 2_feature_engineering
│   ├── feature_engineering_lufthansa.ipynb
│   ├── feature_engineering_singapore.ipynb
│   ├── Google_vision_API.ipynb
│   ├── Google_vision_API_has_airplane.ipynb
│   └── features_EDA.ipynb
├── 3_modelling
│   ├── Autogluon_lufthansa.ipynb
│   ├── Autogluon_singaporeair.ipynb
│   ├── Modelling_Lufthansa_final.ipynb
│   └── Modelling_Singapore_final.ipynb
├── additional_data
├── data
│   ├── lufthansa_usa
│   ├── singaporeair
│   └── photos_training_ML_has_airplane
├── processed_data
│   ├── lufthansa
│   │   ├── basetable_pictures_luf_brand.csv
│   │   └── merged_basetable_luf.csv
│   └── singapore
│       ├── basetable_pictures_sin_brand.csv
│       └── merged_basetable_sin.csv
├── README.md
└── .gitignore
```

# Results and evaluation
Using the prepared data, we constructed predictive models, fine-tuning hyperparameters on the validation set and assessing performance on the test set. Table 2 shows an overview of the top-performing models. 

| Model                        | Test RMSE |
| ---------------------------- | -------------- |
| Linear Regression            |0.1112|
| Decision tree           | 0.0953|
| Gradient Boosting            | 0.0933         |
| Random Forest           | **0.0911**        |
| Autogluon (benchmark)         | **0.0817**        |

**Table 2:** Model Validation Metrics

To determine our most important drivers, we calculated Shapley values on our Autogluon model for Lufthansa (Figure 1)

![Shapley Results](./additional_data/SHAP_lufthansa.png)

**Figure 1:** Shapley Values Lufthansa (Autogluon model)


# Conclusion

Our approach leverages machine learning models to identify the drivers of a brand's engagement, addressing the significant challenges companies face in having to maintain customer relationships. The results (Table 2) highlight the superior predictive accuracy of **Linear Regression**, which is especially attractive given Occam's razor, which states that no more assumptions should be made than necessary. Using these, we found that the main drivers of customer engagement are: **contains airplane**, **the number of tags**, **contains brand logo** and **colourfulness of the picture**. Hence, we recommend to post more pictures with a visible brand logo and plane, to explicitly tag users to brand posts when highlighting their work (“Photo By”…) and to incorporate less colour as to not overwhelm your audience. These findings provide valuable insights, guiding companies toward more informed and data-driven decisions in managing customer engagement.

# Future work

While the current models provide a solid foundation for predicting the drivers of customer engagement, there are several avenues for future exploration and improvement:
- Examine various social media platforms such as TikTok, Twitter, and Instagram Stories/Reels, as there is a noticeable uptick in usage on these channels. This will provide a more comprehensive and precise representation of customer dynamics.
- Evaluate influencer posts (earned shared media) and assess their influence on the performance of your brand's posts.

# Contributors

- Ballegeer Matteo (Email: matteo.ballegeer@ugent.be, Student ID: 01900129)
- De Rocker Yaël (Email: yael.derocker@ugent.be, Student ID: 01904043)
- Janssens Wannes (Email: wanjanss.janssens@ugent.be, Student ID: 01905583)
- Peire Julian (Email: julian.peire@ugent.be, Student ID: 01900199)
- Willemen Tom (Email: tom.willemen@ugent.be, Student ID: 01900194)



# References
[1]	De Vries, L., S. Gensler, and P.S. Leeflang, Popularity of brand posts on brand fan pages: An investigation of the effects of social media marketing. Journal of interactive marketing, 2012. 26(2): p. 83-91.

[2] Hotz, N., What is CRISP DM? Jan. 2023. url: https://www.datascience-pm.com/crisp-dm-2/.

[3]	Meire, M., et al., The role of marketer-generated content in customer engagement marketing. Journal of Marketing, 2019. 83(6): p. 21-42.

[4]	Pansari, A. and V. Kumar, Customer engagement: the construct, antecedents, and consequences. Journal of the Academy of Marketing Science, 2017. 45: p. 294-311.





