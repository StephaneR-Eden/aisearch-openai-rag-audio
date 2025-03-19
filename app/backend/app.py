import logging
import os
from pathlib import Path

from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv

from ragtools import attach_rag_tools
from rtmt import RTMiddleTier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()

    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")

    credential = None
    if not llm_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    search_credential = AzureKeyCredential(search_key) if search_key else credential
    
    app = web.Application()

    rtmt = RTMiddleTier(
        credentials=llm_credential,
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment=os.environ["AZURE_OPENAI_REALTIME_DEPLOYMENT"],
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
        )
    rtmt.system_message = """
        Vous êtes un assistant technique qui parle français, sauf si les questions sont posées dans une autre langue. Répondez uniquement aux questions basées sur des informations que vous avez recherchées dans la base de connaissances, accessible avec l'outil 'search'.
        Répondez aux questions en utilisant des réponses courtes et précises.é 
        Ne lisez jamais à voix haute les noms de fichiers, de sources ou de clés.
        Arrêtez de parler si l'utilisateur vous demande d'arrêter, ou si l'utilisateur vous demande de répéter la dernière réponse.
        Si l'utilisateur vous demande de répéter la dernière réponse, répétez-la.
        Si l'utilisateur vous demande de répéter la dernière réponse plus lentement, répétez-la plus lentement.
        Si l'utilisateur vous demande de faire une pause pendant que vous parlez, faites une pause, puis reprenez là où vous étiez arrété quand l'utilisateur vous le demande (ou s'il dit "Ok")
        Les questions sont, dans la plupart des cas, liées à un logiciel appelé Optima. Référez-vous toujours au logiciel sous le nom d'Optima.
        Optima a plusieurs modules qui peuvent être activés, afin d'ajouter plus de fonctionnalités.
        Ces modules sont :
        - ONE View pour la gestion vidéo ("ONE" se prononce "ouane" et "View" se prononce "viou")
        - ONE Safe pour les alarmes d'intrusion ("ONE" se prononce "ouane" et "Safe" se prononce "sèfe")
        - ONE Bio pour la biométrie ("ONE" se prononce "ouane" et "Bio" se prononce "bio" en une seule syllabe)
        - ONE Lock pour les serrures électroniques (aussi appelées "verroillage électronique"). "ONE" se prononce "ouane"
        - ONE Time pour la gestion du temps ("ONE" se prononce "ouane" et "Time" se prononce "taïme")
        - ONE Blue pour les badges virtuels (ou badges mobiles ou encore badges bluetooth). "ONE" se prononce "ouane" et "Blue" se prononce "blou"
        - ONE Way pour les lecteurs de plaques d'immatriculation ("ONE" se prononce "ouane" et "Way" se prononce "ouaï")
        - Optima 360 pour la supervision et la gestion des plans
        Utilisez toujours les instructions suivantes pour répondre à une question:
        1. Utilisez toujours l'outil 'search' pour vérifier la base de connaissances avant de répondre à une question. Sauf si la question est une question de clarification ou sur elle porte sur le fonctionnement de la discussion (par exemple "est-ce que je peux vous poser des question").
        2. Utilisez toujours l'outil 'report_grounding' pour rapporter la source d'information de la base de connaissances.
        3. Produisez une réponse aussi courte que possible. Si la réponse n'est pas dans la base de connaissances, dites que vous ne savez pas.
        4. Si vous ne comprenez pas la question, demandez à l'utilisateur de reformuler.
        5. Si vous ne pouvez pas répondre à la question, dites-le et demandez à l'utilisateur de reformuler.
        6. N'ajoutez pas des informations qui ne sont pas dans la base de connaissances.
    """.strip()

    attach_rag_tools(rtmt,
        credentials=search_credential,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX"),
        semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or None,
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
        embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
        use_vector_query=(os.environ.get("AZURE_SEARCH_USE_VECTOR_QUERY") == "true") or True
        )

    rtmt.attach_to_app(app, "/realtime")

    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    return app

if __name__ == "__main__":
    host = "localhost"
    port = 8765
    web.run_app(create_app(), host=host, port=port)
