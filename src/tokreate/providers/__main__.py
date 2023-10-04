from .base import ProviderRegistry


def main():
    for model in ProviderRegistry.all():
        print(model)


if __name__ == "__main__":
    main()
