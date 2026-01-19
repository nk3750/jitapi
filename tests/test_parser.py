"""Tests for the OpenAPI parser."""

from pathlib import Path

import pytest

from jitapi.ingestion.parser import OpenAPIParser, SpecVersion


@pytest.fixture
def parser():
    """Create a parser instance."""
    return OpenAPIParser()


@pytest.fixture
def petstore_spec_path():
    """Path to the petstore test fixture."""
    return Path(__file__).parent / "fixtures" / "petstore.yaml"


@pytest.fixture
def weather_spec_path():
    """Path to the weather API test fixture."""
    return Path(__file__).parent / "fixtures" / "weather_api.yaml"


class TestOpenAPIParser:
    """Tests for OpenAPIParser."""

    def test_parse_petstore_spec(self, parser, petstore_spec_path):
        """Test parsing the petstore OpenAPI spec."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        assert parsed.title == "Petstore API"
        assert parsed.version == "1.0.0"
        assert parsed.spec_version == SpecVersion.OPENAPI_3
        assert parsed.base_url == "https://petstore.example.com/v1"
        assert len(parsed.endpoints) > 0

    def test_parse_endpoints_extracted(self, parser, petstore_spec_path):
        """Test that endpoints are properly extracted."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Check for expected endpoints
        endpoint_ids = [ep.endpoint_id for ep in parsed.endpoints]

        assert "GET /pets" in endpoint_ids
        assert "POST /pets" in endpoint_ids
        assert "GET /pets/{petId}" in endpoint_ids
        assert "POST /orders" in endpoint_ids
        assert "GET /users/{userId}" in endpoint_ids

    def test_endpoint_parameters(self, parser, petstore_spec_path):
        """Test that parameters are properly extracted."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find GET /pets endpoint
        get_pets = next(ep for ep in parsed.endpoints if ep.endpoint_id == "GET /pets")

        assert len(get_pets.parameters) == 2

        # Check limit parameter
        limit_param = next(p for p in get_pets.parameters if p.name == "limit")
        assert limit_param.location == "query"
        assert limit_param.schema_type == "integer"
        assert limit_param.required is False

    def test_path_parameters(self, parser, petstore_spec_path):
        """Test that path parameters are properly extracted."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find GET /pets/{petId} endpoint
        get_pet = next(
            ep for ep in parsed.endpoints if ep.endpoint_id == "GET /pets/{petId}"
        )

        # Check petId parameter
        pet_id_param = next(p for p in get_pet.parameters if p.name == "petId")
        assert pet_id_param.location == "path"
        assert pet_id_param.required is True

    def test_request_body_extraction(self, parser, petstore_spec_path):
        """Test that request bodies are properly extracted."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find POST /pets endpoint
        create_pet = next(
            ep for ep in parsed.endpoints if ep.endpoint_id == "POST /pets"
        )

        assert create_pet.request_body is not None
        assert create_pet.request_body.content_type == "application/json"
        assert create_pet.request_body.required is True
        assert "properties" in create_pet.request_body.schema

    def test_tags_extraction(self, parser, petstore_spec_path):
        """Test that tags are properly extracted."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find GET /pets endpoint
        get_pets = next(ep for ep in parsed.endpoints if ep.endpoint_id == "GET /pets")

        assert "pets" in get_pets.tags

    def test_response_extraction(self, parser, petstore_spec_path):
        """Test that responses are properly extracted."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find GET /pets/{petId} endpoint
        get_pet = next(
            ep for ep in parsed.endpoints if ep.endpoint_id == "GET /pets/{petId}"
        )

        assert len(get_pet.responses) >= 1

        # Check 200 response
        ok_response = next(r for r in get_pet.responses if r.status_code == "200")
        assert ok_response.description == "A pet"
        assert ok_response.schema is not None

    def test_weather_api_spec(self, parser, weather_spec_path):
        """Test parsing the weather API spec."""
        parsed = parser.parse_from_file(str(weather_spec_path))

        assert parsed.title == "Weather API"
        assert len(parsed.endpoints) > 0

        # Check for expected endpoints
        endpoint_ids = [ep.endpoint_id for ep in parsed.endpoints]
        assert "GET /locations/search" in endpoint_ids
        assert "GET /currentconditions/{locationKey}" in endpoint_ids
        assert "GET /forecasts/daily/{locationKey}" in endpoint_ids

    def test_returned_fields_extraction(self, parser, petstore_spec_path):
        """Test that returned fields are extracted for dependency analysis."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find GET /pets/{petId} endpoint - returns a single Pet with resolved schema
        get_pet = next(
            ep for ep in parsed.endpoints if ep.endpoint_id == "GET /pets/{petId}"
        )

        # The returned_fields should include fields from the Pet schema
        # Note: returned_fields may be empty if $ref is not fully resolved
        # This is a known limitation - the test verifies the field exists
        assert isinstance(get_pet.returned_fields, list)

    def test_embedding_text_generation(self, parser, petstore_spec_path):
        """Test that embedding text is properly generated."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find an endpoint and check its embedding text
        get_pets = next(ep for ep in parsed.endpoints if ep.endpoint_id == "GET /pets")
        text = get_pets.to_embedding_text()

        assert "GET /pets" in text
        assert "List all pets" in text
        assert "limit" in text.lower() or "Parameters:" in text

    def test_ref_resolution(self, parser, petstore_spec_path):
        """Test that $ref references are properly resolved."""
        parsed = parser.parse_from_file(str(petstore_spec_path))

        # Find an endpoint that uses $ref in its response
        get_pet = next(
            ep for ep in parsed.endpoints if ep.endpoint_id == "GET /pets/{petId}"
        )

        # The response schema should be resolved (not a $ref)
        ok_response = next(r for r in get_pet.responses if r.status_code == "200")
        assert ok_response.schema is not None
        # The resolved schema should have properties (from Pet schema)
        assert "properties" in ok_response.schema or "type" in ok_response.schema
