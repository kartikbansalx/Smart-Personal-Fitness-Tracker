# Smart Personal Fitness Tracker

A machine learning-powered web application for personalized fitness analytics and calorie burn prediction. This project helps users track, compare, and predict calories burned during exercise using physiological and workout data.

## Features

- **Calorie Burn Prediction:** Input your age, BMI, exercise duration, heart rate, body temperature, and gender to receive a real-time prediction of calories burned.
- **Visual Insights:** View comparison charts for calories burned, heart rate, body temperature, and exercise duration against other users.
- **Peer Group Comparison:** See how your personal stats rank compared to a dataset of similar users.
- **Similar Case Samples:** Get sample data from users with similar exercise profiles.
- **Streamlit Web App:** Interactive user interface for quick data entry and instant results.

## Technologies Used

- **Python**
- **Streamlit** for the web interface
- **scikit-learn** for machine learning (Random Forest regression)
- **pandas** and **numpy** for data manipulation
- **matplotlib** and **seaborn** for data visualization

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kartikbansalx/Smart-Personal-Fitness-Tracker.git
   cd Smart-Personal-Fitness-Tracker
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Prepare the data:**
   - Ensure you have the files `calories.csv` and `exercise.csv` in the project root. These files should contain exercise and physiological data.

4. **Run the app:**
   ```bash
   streamlit run appfinal.py
   ```
   *You can also try `app.py` or `appnew.py` for alternate versions.*

## Usage

1. Launch the app using the command above.
2. Enter your data in the sidebar:
   - Age
   - BMI
   - Duration (minutes)
   - Heart Rate
   - Body Temperature (°C)
   - Gender
3. View your predicted calories burned, compare your stats with peers, and explore users with similar exercise profiles.

## How It Works

- The app loads and merges user data from `calories.csv` and `exercise.csv`.
- BMI is calculated automatically.
- A Random Forest regression model is trained on exercise data to predict calories burned.
- User input is aligned with model features for accurate prediction.
- Comparison charts and peer analysis are generated for insights.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

## License

This project currently does not specify a license.

## Acknowledgements

Developed by **Kartik Bansal**  
Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).

---

Feel free to open issues for bug reports, feature requests, or questions!
