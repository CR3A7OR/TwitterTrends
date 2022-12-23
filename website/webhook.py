from config import *

# Function which sends a webhook to a channel listed as the URL variable
def discord_webhook():
    
    # Building a webhook payload with a bot name and avatar
    url = "DISCORD WEBHOOK URL"
    username ="TwiTR"
    avatar_url = "https://imgur.com/NkF5oei.png"
    webhook = DiscordWebhook(url=url, username=username, avatar_url=avatar_url)

    with open("./static/img/RewindLogoSmall.png", "rb") as f:
        webhook.add_file(file=f.read(), filename='logo.png')

    with open("./static/img/twitter.png", "rb") as f:
        webhook.add_file(file=f.read(), filename='twit.png')

    # create embed object for webhook (URL NEEDS TO BE CHANGED ONCE SITE IS UP)
    embed = DiscordEmbed(title='TRENDING IN UK TWITTER', description='These are the current top 5 trends on twitter within the UK along with the most recent tweet', color='00aced', url='https://github.com/CR3A7OR/MENG_GP')

    # set image
    embed.set_image(url='attachment://twit.png')

    # set thumbnail
    embed.set_thumbnail(url='attachment://logo.png')
    embed.set_footer(text='Updated Daily | View the article online', icon_url=avatar_url)
    embed.set_timestamp()

    # add fields to embed by looping through all trends in the JSON file
    f = open('output.json')
    data = json.load(f)
    for trend in data['trends']:
        if(len(trend['name']) + len(trend['article']) + len(trend['url']) < 996):
            embed.add_embed_field(name=trend['title'], value= "*{}*\n {} **[Follow the trend]({})**".format(trend['name'],trend['article'], trend['url']), inline=False)
        else:
            remove = len(trend['name']) + len(trend['article']) + len(trend['url']) - 995
            embed.add_embed_field(name=trend['title'], value= "*{}*\n {} **[Follow the trend]({})**".format(trend['name'],trend['article'][:-remove], trend['url']), inline=False)
    # add embed object to webhook and send it
    webhook.add_embed(embed)
    response = webhook.execute(remove_embeds=True)

discord_webhook()