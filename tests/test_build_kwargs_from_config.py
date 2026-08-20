"""
Test de validation pour build_kwargs_from_config (Story 12.1, AD-21) :
introspection centralisee qui construit les kwargs d'une factory modele/classe
Strategy depuis deux canaux distincts - config_keys (inconditionnel) et
overrides (strict, jamais absorbe par un **kwargs catch-all). Remplace les
anciennes branches `if "X" in config: model_kwargs["X"] = config["X"]`
dupliquees dans main.py.

Script autonome - meme convention que les autres tests de ce projet (pas de
framework de test formel impose). Executer directement :
    python tests/test_build_kwargs_from_config.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_library import build_kwargs_from_config


def _fake_named(a, b=1):
    pass


def _fake_kwargs(a, **kw):
    pass


def _fake_default(a, b=5):
    pass


def test_override_forwarded_only_if_named_explicitly():
    """AC #1 : une cle d'overrides absente de la signature nommee (target n'a
    qu'un **kwargs catch-all) n'est jamais forwardee - discipline stricte
    heritee d'AD-3 (architecture-compute-dtype-hardware-2026-08-17)."""
    kwargs, forwarded = build_kwargs_from_config(_fake_kwargs, {}, a=10, c=20)
    assert kwargs == {"a": 10}, kwargs
    assert forwarded == frozenset({"a"}), forwarded
    assert "c" not in kwargs
    print("OK - override non nomme explicitement (absorbe par **kwargs) jamais forwarde")


def test_config_keys_forwarded_unconditionally_even_via_kwargs():
    """AC #2 : une cle de config_keys presente dans config EST forwardee, que
    la cible la nomme explicitement ou l'absorbe via **kwargs - preserve le
    comportement reel actuel des 5 factories create_chess_*/
    create_aircraft_detector_unet."""
    kwargs, forwarded = build_kwargs_from_config(
        _fake_kwargs, {"a": 1, "z": 2}, config_keys=("a", "z")
    )
    assert kwargs == {"a": 1, "z": 2}, kwargs
    assert forwarded == frozenset(), forwarded
    print("OK - cles config_keys forwardees sans condition, meme via **kwargs")


def test_overrides_wins_over_config_key_of_same_name():
    """AC #3 : une cle presente a la fois dans config_keys/config et dans
    overrides -> overrides gagne silencieusement, contrat documente."""
    kwargs, forwarded = build_kwargs_from_config(
        _fake_named, {"a": 1}, config_keys=("a",), a=2
    )
    assert kwargs == {"a": 2}, kwargs
    assert forwarded == frozenset({"a"}), forwarded
    print("OK - overrides gagne sur une cle config de meme nom")


def test_missing_key_never_filled_with_helper_chosen_default():
    """Une cle nommee dans la signature avec un defaut, absente de config et
    d'overrides, n'est jamais comblee par le helper - le dict retourne ne
    contient pas la cle, le defaut du constructeur cible s'applique en aval."""
    kwargs, forwarded = build_kwargs_from_config(_fake_default, {}, config_keys=())
    assert kwargs == {}, kwargs
    assert forwarded == frozenset(), forwarded
    print("OK - cle absente des deux sources jamais comblee par un defaut du helper")


def test_target_none_returns_empty_kwargs():
    """target peut etre None (ex. MODELS.get(model_name) sur un nom invalide)
    - ne doit pas planter ici, get_model() leve son ValueError explicite en aval."""
    kwargs, forwarded = build_kwargs_from_config(None, {"a": 1}, config_keys=("a",), compute_dtype="x")
    assert kwargs == {}, kwargs
    assert forwarded == frozenset(), forwarded
    print("OK - target=None retourne des kwargs vides sans planter")


def test_config_keys_absent_from_signature_still_forwarded():
    """Cle de config_keys absente ET de la signature nommee ET d'un **kwargs
    catch-all (target n'a ni l'un ni l'autre) - toujours forwardee (canal
    inconditionnel, meme comportement que l'ancien `if "X" in config`, qui ne
    verifiait pas non plus la signature cible avant d'ajouter la cle)."""
    kwargs, forwarded = build_kwargs_from_config(
        _fake_named, {"unrelated": 1}, config_keys=("unrelated",)
    )
    assert kwargs == {"unrelated": 1}, kwargs
    print("OK - canal config_keys ne verifie pas la signature (comportement identique a l'existant)")


if __name__ == "__main__":
    test_override_forwarded_only_if_named_explicitly()
    test_config_keys_forwarded_unconditionally_even_via_kwargs()
    test_overrides_wins_over_config_key_of_same_name()
    test_missing_key_never_filled_with_helper_chosen_default()
    test_target_none_returns_empty_kwargs()
    test_config_keys_absent_from_signature_still_forwarded()
