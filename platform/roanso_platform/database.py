# -*- coding: utf-8 -*-
"""Gestión de Motor SQLAlchemy 2.0 (PostgreSQL Pool + SQLite WAL) y Semillas Éticas."""
from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, VoiceProfile

CATALOG_VOICES_SEED = [
    {
        "id": "catalog-chileno-sereno-m1",
        "display_name": "Narrador Sureño Sereno (Catálogo Ético es-CL)",
        "locale": "es-CL",
        "storage_key": "catalog/chileno_sereno_m1.pt",
    },
    {
        "id": "catalog-chilena-calida-f1",
        "display_name": "Narradora Valle Central Cálida (Catálogo Ético es-CL)",
        "locale": "es-CL",
        "storage_key": "catalog/chilena_calida_f1.pt",
    },
]


def build_engine(database_url: str) -> Engine:
    is_sqlite = database_url.startswith("sqlite")
    kwargs: dict[str, Any] = {"future": True}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20

    engine = create_engine(database_url, **kwargs)

    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_conn: Any, _record: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()

    return engine


def init_db_and_seed_catalog(engine: Engine) -> sessionmaker[Session]:
    """Inicializa tablas y registra el catálogo base de voces éticas licenciadas."""
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    with session_factory() as db:
        for item in CATALOG_VOICES_SEED:
            exists = db.scalar(select(VoiceProfile).where(VoiceProfile.id == item["id"]))
            if not exists:
                db.add(
                    VoiceProfile(
                        id=item["id"],
                        family_id=None,
                        display_name=item["display_name"],
                        is_catalog=True,
                        locale=item["locale"],
                        storage_key=item["storage_key"],
                        status="ACTIVE",
                    )
                )
        db.commit()
    return session_factory
