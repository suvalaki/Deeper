from __future__ import annotations
from sqlalchemy.orm import sessionmaker, scoped_session
import logging

logger = logging.getLogger(__name__)

## \addtogroup Loaders
# I am getting added to Loaders page
# \{


class SqlAlchemySessionGetter:
    """! Base class to provide threadsafe scoped_sessions to sqlite data getters.

    Example of how to subclass type to create a data loader automatically.
    
    
    class SqliteLoader(ChildOf_DataExtractor, SqlAlchemySessionGetter):

        def __init__(self, engine=None, session=None):
            super().__init__(engine, session)

        def get(self, seasons: Sequence[int] = None):
            return query_with_session_all_player_data(self._session, seasons)
    
    
    """

    def __init__(self, engine=None, session=None, **kwargs):
        """! Initialiser
        @param engine sqlalchemy engine pointint to the sql database. Takes
            precedence over provided session.
        @param session sqlalchemy session (TODO: REMOVE THIS)
        """

        if engine == None and session == None:
            logger.error("No engine or session provided. )

        self._engine = engine
        self._session = engine
        if engine is None:
            self._session = session
        else:
            self._Sessionmaker = scoped_session(sessionmaker(bind=engine))
            self._session = self._Sessionmaker()

    @classmethod
    def from_session(cls, session) -> SqlAlchemySessionGetter:
        return cls(None, session)

    # @classmethod
    # def from_default_engine(cls) -> SqlAlchemySessionGetter:
    #     """! Construct a new SqlalchemySessionGetter from the default sqllite
    #     engine location
    #     @details The default engine presumes that the databse is stored as a
    #     sqlite file at `./brownlowdb.db`.
    #     """
    #     session = get_engine()
    #     return cls(None, session)

    def close_connections(self):
        self._session.close()
        self._session = None
        self._Sessionamker = None
        for k, v in self.__dict__.items():
            if isinstance(v, SqlAlchemySessionGetter):
                v.close_connections()


## \}
