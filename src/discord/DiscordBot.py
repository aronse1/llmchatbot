import discord
import functools
import logging
import asyncio
from discord import Message
from discord.ext import commands
from src.ChatBot import ChatBot, Course
from src.discord.Dropdowns import DropdownView
from src.discord.disclaimer import disclaimer
from src.Pipeline import *
from src.discord.MessageManager import *
chatbot_logger = logging.getLogger('ChatBot')


class DiscordBot(commands.Bot):
    #chatbot: ChatBot = None

    def __init__(self, documents_dir: str, index_dir: str):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_reactions = True
        super().__init__(command_prefix=commands.when_mentioned_or('$'), intents=intents)
        #self.chatbot = ChatBot(
        #    documents_dir=documents_dir, index_dir=index_dir)
        initialise()

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message: Message):
        # ignore messages from the bot itself
        if message.author.id == self.user.id:
            return

        # only reply to DMs
        if not isinstance(message.channel, discord.channel.DMChannel):
            return

        # get pinned messages
        pinned_messages = await message.channel.pins()

        course: Course = None
        disclaimer_present = False
        # check if course is pinned
        for msg in pinned_messages:
            if msg.content.startswith('Kurs:') and msg.author.id == self.user.id:
                course = Course(msg.content.split(': ')[1].lower())
            if msg.content.startswith("## **Disclaimer:**"):
                disclaimer_present = True

        if not disclaimer_present:
            msg = await message.channel.send(disclaimer)
            await msg.pin()

        if course is None:
            view = DropdownView()
            await message.channel.send('Bitte wähle deinen Kurs und stelle die Frage anschließend erneut', view=view)
        else:
            sent_message = await message.channel.send("Thinking...")
            #fun = functools.partial(
            #   self.chatbot.perform_query, message.content, course)
            
            #response = await self.loop.run_in_executor(None, fun)
            c = AdvancedRAGWorkflow2(timeout=3600, verbose=True, course=course, userid=message.author.id)
            #try:
            response = await c.run(query=message.content)
            # except:
            #     response = "Das kann ich leider nicht Beantworten"
            await sent_message.edit(content=response)
            await save_message(message.author.id, "user", message.content)
            await save_message(message.author.id, "assistant", str(response))

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id == self.user.id:
            return
        channel = self.get_channel(payload.channel_id)
        

        if channel is None:
           
            user = self.get_user(payload.user_id)
            if not user:
                try:
                    user = await self.fetch_user(payload.user_id)
                except discord.NotFound:
                    return
            channel = user.dm_channel
            
            if channel is None:
                channel = await user.create_dm()
        
        try:
            message = await channel.fetch_message(payload.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            print(f"Konnte Nachricht {payload.message_id} nicht abrufen")
            return
        
        if message.author.id != self.user.id:
            return
    
        print(f"Reaktion erkannt: {payload.emoji.name} von {payload.user_id} in Channel {channel.id}")
    
        if payload.emoji.name == "👍":
            await clear_history(payload.user_id)
            await channel.send(f"Neuer Chat für dich, <@{payload.user_id}>")