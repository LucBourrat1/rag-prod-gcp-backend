from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.auth import get_current_user
from app.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
def authenticated_user():
    app.dependency_overrides[get_current_user] = lambda: {"email": "test@test.com"}
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_documents_sans_auth(client):
    response = await client.get("/documents")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


@pytest.mark.asyncio
async def test_upload_sans_auth(client):
    response = await client.post("/documents/upload")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_upload_fichier_invalide(client, authenticated_user):
    files = {"file": ("document.txt", b"contenu", "text/plain")}
    response = await client.post("/documents/upload", files=files)
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_sans_documents(client, authenticated_user):
    with patch("app.main.ask_question", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {"answer": "Réponse test", "sources": []}
        response = await client.post(
            "/chat",
            json={"question": "Qu'est-ce que ce document ?", "document_ids": []},
        )
        assert response.status_code == 200
        assert "answer" in response.json()
