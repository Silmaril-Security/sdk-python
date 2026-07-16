# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

from silmaril_security.sdk.sanitization import sanitize_text


def test_sanitize_text_removes_lone_surrogates_and_preserves_valid_unicode():
    assert sanitize_text("hello\ud800world\udc00") == "helloworld"
    assert sanitize_text("developer 👩‍💻 ready") == "developer 👩‍💻 ready"
