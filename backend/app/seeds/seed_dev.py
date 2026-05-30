import asyncio, os
from app.database import AsyncSessionLocal
from app.schemas.account import AccountCreate
from app.services.account_service import AccountService
from app.services.notebook_inventory import NotebookInventory

async def main():
    print({'notebooks': NotebookInventory().list()})
    if os.getenv('DEV_KAGGLE_ACCOUNT_NAME') and os.getenv('DEV_KAGGLE_USERNAME') and os.getenv('DEV_KAGGLE_KEY'):
        async with AsyncSessionLocal() as s:
            svc=AccountService(s)
            await svc.create_account(AccountCreate(name=os.environ['DEV_KAGGLE_ACCOUNT_NAME'], kaggle_username=os.environ['DEV_KAGGLE_USERNAME'], kaggle_key=os.environ['DEV_KAGGLE_KEY']))
            print('seeded dev account')
    else:
        print('no dev Kaggle account seeded')
if __name__ == '__main__': asyncio.run(main())
