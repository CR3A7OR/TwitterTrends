<p align="center" width="100%">
    <img width="60%" src="/website/static/img/RewindLogoSmall.png">
</p>
<div align="center">
  <h4>│ Your Daily Catchup │</h4>
</div>

An AI-powered social media summation project primarily focused on the use of Twitter’s available API to collect a daily roundup of top 5 trending topics within the UK, along with a collected sample of tweets categorised under the relevant trend. Analysing these tweets it generates a digestible overview of the trending event to be expressed in the form of a mini-news article. Distributed using dynamic content generation through Model-View templates across integrated platforms to read the results. *(The project is not perfect and mostly stands as a proof of concept within the proposal of automation for AI driven article generation)*

| Linux  | Windows |
|--------|---------|
| ![GitHub Workflow Status](https://github.com/CR3A7OR/AutoSleuth/blob/main/README_Photos/Linux%20passing.svg) | ![GitHub Workflow Status](https://github.com/CR3A7OR/AutoSleuth/blob/main/README_Photos/Windows%20passing.svg) |

## »│ Technical Breakdown

#### │ **Sentiment**:
> - Twitter's API is used to snapshot the current top `5 trends` from the UK, along with `70 of the most recent tweets` for them
> - A `cleaned word list` of top used words is generated from the snapshot for each trend
> - `Sentiment Score` is calculated for each tweet through an average between logistic regression and an open-source tool called [Vader](https://github.com/cjhutto/vaderSentiment)

#### │ **Text Generation**: 
> - `Google’s index` for news is then scraped for a minimum of 3 best fitting `headlines` within a 5 hour time frame, which are scored based on containing
the top 3 most commonly referred to nouns and adjectives in our tweet set and expressing a matching sentiment
> - `Nouns` and `adjectives` from the headlines are extracted and used to generate our cohesive seed sentence using `T5`, a transformer based architecture that uses a text-to-text approach *(the headline for the article)*
> - The `seed sentence` acts as the input being fed into `GPT-2` for large text generation *(the article)*

#### │ **User Interface**: 
> - Module saves these outputs and all relevant data to `output.json` including: 
>   - trend │ headline │ article │ twitter link │ image
> - `Python Flask` is our web framework and using the Model-View Template we dynamically populate CSS containers on the website
> - `Flask Mail` is used for sending emails to all user's in `emails.csv` populated through the website
> - `Discord-Webhook` used for sending HTTP POST requests to listening Discord endpoints

```diff
- THE PROJECT WEBSITE IS AVAILABLE BELOW BUT IS STATIC AS OF NOW DUE TO SERVER ARCHITECTURE LIMITATIONS -
```
 
## »│ Operartion

<table>
    <tr>
      <td><p align="center">Email</p></td>
      <td><p align="center">Website</p></td>
      <td><p align="center">Discord</p></td>
    </tr>
    <tr>
      <td><img src="/READMEIMG/email.png" /></td>
      <td><img src="/READMEIMG/web.png" /> </td>
      <td><img src="/READMEIMG/discordEmbed.png" /> </td>
    </tr>
</table>

Above is a live view of how the system interface is designed, the website can be found at [**TwitterTrends**](http://twitr.trioffline.com/)

## »│ Developers
Thank you to the following people:
<table>
  <tr>
      <td align="center"><a href="https://github.com/Ollyentwistle"><img src="https://avatars.githubusercontent.com/u/67477209?v=4" width="100px;" alt=""/><br /><sub><b>CR3A7OR</b></sub></a><br /><a href="#design-CR3A7OR" title="Design">🎨</a> <a href="#maintenance-CR3A7OR" title="Maintenance">🚧</a> <a href="WEB-CR3A7OR" title="Code">💻</a> <a href="#ideas-CR3A7OR" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/TolGb01"><img src="https://avatars1.githubusercontent.com/u/50290936?v=4?s=100" width="100px;" alt=""/><br /><sub><b>TolGb01</b></sub></a><br /><a href="#maintenance-Tolu" title="Maintenance">🚧</a> <a href="AI-Tolu" title="Code">💻</a> <a href="#ideas-Tolu" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/Ollyentwistle"><img src="https://avatars.githubusercontent.com/u/61790858?v=4" width="100px;" alt=""/><br /><sub><b>Ollyentwistle</b></sub></a><br /><a href="Sentiment-Olly" title="Code">💻</a> <a href="#ideas-Olly" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>
